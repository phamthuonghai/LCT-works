#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self, args, source_chars, target_chars, bow, eow, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.source_ids = tf.placeholder(tf.int32, [None, None], name="source_ids")
            self.source_seqs = tf.placeholder(tf.int32, [None, None], name="source_seqs")
            self.source_seq_lens = tf.placeholder(tf.int32, [None], name="source_seq_lens")
            self.target_ids = tf.placeholder(tf.int32, [None, None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            # Training. The rest of the code assumes that
            # - when training the decoder, the output layer with logits for each generated
            #   character is in `output_layer` and the corresponding predictions are in
            #   `self.predictions_training`.
            # - the `target_ids` contains the gold generated characters
            # - the `target_lens` contains number of valid characters for each lemma
            # - when running decoder inference, the predictions are in `self.predictions`
            #   and their lengths in `self.prediction_lens`.
            output_layer, self.predictions_training, target_ids, target_lens, self.predictions, self.prediction_lens = \
                self.build_model(args, source_chars, target_chars, bow, eow)

            # Training
            weights = tf.sequence_mask(target_lens, dtype=tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(target_ids, output_layer, weights=weights)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            accuracy_training = tf.reduce_all(tf.logical_or(
                tf.equal(self.predictions_training, target_ids),
                tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
            self.current_accuracy_training, self.update_accuracy_training = tf.metrics.mean(accuracy_training)

            minimum_length = tf.minimum(tf.shape(self.predictions)[1], tf.shape(target_ids)[1])
            accuracy = tf.logical_and(
                tf.equal(self.prediction_lens, target_lens),
                tf.reduce_all(tf.logical_or(
                    tf.equal(self.predictions[:, :minimum_length], target_ids[:, :minimum_length]),
                    tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
            self.current_accuracy, self.update_accuracy = tf.metrics.mean(accuracy)

            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/lr", self.learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy_training)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def build_model(self, args, source_chars, target_chars, bow, eow):
        # Append EOW after target_seqs
        target_seqs = tf.reverse_sequence(self.target_seqs, self.target_seq_lens, 1)
        target_seqs = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=eow)
        target_seq_lens = self.target_seq_lens + 1
        target_seqs = tf.reverse_sequence(target_seqs, target_seq_lens, 1)

        # Encoder
        # Generate source embeddings for source chars, of shape [source_chars, args.char_dim].
        source_embedding = tf.get_variable('source_embedding', [source_chars, args.char_dim])

        # Embed the self.source_seqs using the source embeddings.
        source_embedded = tf.nn.embedding_lookup(source_embedding, self.source_seqs)

        # Using a GRU with dimension args.rnn_dim, process the embedded self.source_seqs
        # using bidirectional RNN. Store the summed fwd and bwd outputs in `source_encoded`
        # and the summed fwd and bwd states into `source_states`.
        source_rnn_outputs, source_rnn_states = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(args.rnn_dim), tf.nn.rnn_cell.GRUCell(args.rnn_dim),
            source_embedded, sequence_length=self.source_seq_lens, dtype=tf.float32
        )
        source_encoded = source_rnn_outputs[0] + source_rnn_outputs[1]
        source_states = source_rnn_states[0] + source_rnn_states[1]

        # Index the unique words using self.source_ids and self.target_ids.
        sentence_mask = tf.sequence_mask(self.sentence_lens)
        source_encoded = tf.boolean_mask(tf.nn.embedding_lookup(source_encoded, self.source_ids), sentence_mask)
        source_states = tf.boolean_mask(tf.nn.embedding_lookup(source_states, self.source_ids), sentence_mask)
        source_lens = tf.boolean_mask(tf.nn.embedding_lookup(self.source_seq_lens, self.source_ids), sentence_mask)

        target_seqs = tf.boolean_mask(tf.nn.embedding_lookup(target_seqs, self.target_ids), sentence_mask)
        target_lens = tf.boolean_mask(tf.nn.embedding_lookup(target_seq_lens, self.target_ids), sentence_mask)

        # Decoder
        # Generate target embeddings for target chars, of shape [target_chars, args.char_dim].
        target_embedding = tf.get_variable('target_embedding', [target_chars, args.char_dim])

        # Embed the target_seqs using the target embeddings.
        target_embedded = tf.nn.embedding_lookup(target_embedding, target_seqs)

        # Generate a decoder GRU with dimension args.rnn_dim.
        gru_decoder = tf.nn.rnn_cell.GRUCell(args.rnn_dim)

        # Create a `decoder_layer` -- a fully connected layer with
        # target_chars neurons used in the decoder to classify into target characters.
        decoder_layer = tf.layers.Dense(target_chars)

        # Attention
        # Generate three fully connected layers without activations:
        # - `source_layer` with args.rnn_dim units
        # - `state_layer` with args.rnn_dim units
        # - `weight_layer` with 1 unit
        source_layer = tf.layers.Dense(args.rnn_dim)
        state_layer = tf.layers.Dense(args.rnn_dim)
        weights_layer = tf.layers.Dense(1)

        def with_attention(inputs, states):
            # Generate the attention

            # Project source_encoded using source_layer.
            source_encoded_projected = source_layer(source_encoded)

            # Change shape of states from [a, b] to [a, 1, b] and project it using state_layer.
            states_ext = state_layer(tf.expand_dims(states, 1))

            # Sum the two above projections, apply tf.tanh and project the result using weight_layer.
            # The result has shape [x, y, 1].
            w = weights_layer(tf.tanh(source_encoded_projected + states_ext))

            # Apply tf.nn.softmax to the latest result, using axis corresponding to source characters.
            w = tf.nn.softmax(w, axis=1)

            # Multiply the source_encoded by the latest result, and sum the results with respect
            # to the axis corresponding to source characters. This is the final attention.
            x = tf.reduce_sum(tf.multiply(w, source_encoded), axis=1)

            # Return concatenation of inputs and the computed attention.
            return tf.concat([inputs, x], axis=-1)

        # The DecoderTraining will be used during training. It will output logits for each
        # target character.
        class DecoderTraining(tf.contrib.seq2seq.Decoder):
            @property
            def batch_size(self):
                # Return size of the batch, using for example source_states size
                return tf.shape(source_states)[0]

            @property
            def output_dtype(self): return tf.float32  # Type for logits of target characters

            @property
            def output_size(self): return target_chars  # Length of logits for every output

            def initialize(self, name=None):
                finished = target_lens <= 0  # False if target_lens > 0, True otherwise
                states = source_states  # Initial decoder state to use
                # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                # You can use tf.fill to generate BOWs of appropriate size.
                inputs = with_attention(
                    tf.nn.embedding_lookup(target_embedding, tf.fill([self.batch_size], bow)),
                    states
                )
                return finished, inputs, states

            def step(self, time, inputs, states, name=None):
                outputs, states = gru_decoder(inputs, states)  # Run the decoder GRU cell using inputs and states.
                outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                # Next input is with_attention called on words with index `time` in target_embedded.
                next_input = with_attention(
                    tf.gather(target_embedded, time, axis=1),
                    states
                )
                finished = target_lens <= time + 1  # False if target_lens > time + 1, True otherwise.
                return outputs, states, next_input, finished

        output_layer, _, _ = tf.contrib.seq2seq.dynamic_decode(DecoderTraining())
        predictions_training = tf.argmax(output_layer, axis=2, output_type=tf.int32)

        # The DecoderPrediction will be used during prediction. It will
        # directly output the predicted target characters.
        class DecoderPrediction(tf.contrib.seq2seq.Decoder):
            @property
            def batch_size(self):
                # Return size of the batch, using for example source_states size
                return tf.shape(source_states)[0]

            @property
            def output_dtype(self): return tf.int32  # Type for predicted target characters

            @property
            def output_size(self): return 1  # Will return just one output

            def initialize(self, name=None):
                finished = tf.fill([self.batch_size], False)  # False of shape [self.batch_size].
                states = source_states  # Initial decoder state to use.
                # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                # You can use tf.fill to generate BOWs of appropriate size.
                inputs = with_attention(
                    tf.nn.embedding_lookup(target_embedding, tf.fill([self.batch_size], bow)),
                    states
                )
                return finished, inputs, states

            def step(self, time, inputs, states, name=None):
                outputs, states = gru_decoder(inputs, states)  # Run the decoder GRU cell using inputs and states.
                outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                # Use tf.argmax to choose most probable class (supply parameter `output_type=tf.int32`).
                outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
                # Embed `outputs` using target_embeddings and pass it to with_attention.
                next_input = with_attention(
                    tf.nn.embedding_lookup(target_embedding, outputs),
                    states
                )
                # True where outputs==eow, False otherwise
                # Use tf.equal for the comparison, Python's '==' is not overloaded
                finished = tf.equal(outputs, eow)
                return outputs, states, next_input, finished

        predictions, _, prediction_lens = tf.contrib.seq2seq.dynamic_decode(
            DecoderPrediction(), maximum_iterations=tf.reduce_max(source_lens) + 10)

        return output_layer, predictions_training, target_seqs, target_lens, predictions, prediction_lens

    def train_epoch(self, train, batch_size, learning_rate):
        while not train.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size,
                                                                                     including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run(
                [self.training, self.summaries["train"]],
                {self.sentence_lens: sentence_lens,
                 self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                 self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                 self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS],
                 self.learning_rate: learning_rate})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                       including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                              self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                              self.source_seq_lens: charseq_lens[train.FORMS],
                              self.target_seq_lens: charseq_lens[train.LEMMAS]})
        acc, loss, _ = self.session.run([self.current_accuracy, self.current_loss, self.summaries[dataset_name]])
        return acc, loss

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)

    def predict(self, dataset, batch_size):
        lemmas = []
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                       including_charseqs=True)
            predictions, prediction_lengths = self.session.run(
                [self.predictions, self.prediction_lens],
                {self.sentence_lens: sentence_lens, self.source_ids: charseq_ids[train.FORMS],
                 self.source_seqs: charseqs[train.FORMS], self.source_seq_lens: charseq_lens[train.FORMS]})

            for length in sentence_lens:
                lemmas.append([])
                for i in range(length):
                    lemmas[-1].append("")
                    for j in range(prediction_lengths[i] - 1):
                        lemmas[-1][-1] += train.factors[train.LEMMAS].alphabet[predictions[i][j]]
                predictions, prediction_lengths = predictions[length:], prediction_lengths[length:]

        return lemmas


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    import sys

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--char_dim", default=256, type=int, help="Character embedding dimension.")
    parser.add_argument("--rnn_dim", default=256, type=int, help="Dimension of the encoder and the decoder.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"):
        os.mkdir("logs")  # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = Network(args, len(train.factors[train.FORMS].alphabet), len(train.factors[train.LEMMAS].alphabet),
                      train.factors[train.LEMMAS].alphabet_map["<bow>"],
                      train.factors[train.LEMMAS].alphabet_map["<eow>"],
                      threads=args.threads)

    # Train
    min_loss = 10000
    early_stopping = 0
    lr_drop_max = 4
    lr_drop_rate = 0.5
    early_stop = 20
    min_learning_rate = 1e-4
    lr = 1e-3
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size, lr)

        accuracy, cur_loss = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
        sys.stdout.flush()
        if cur_loss < min_loss:
            min_loss = cur_loss
            network.save(os.path.join(args.logdir, "model"))
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping % lr_drop_max == 0:
                lr *= lr_drop_rate
                lr = max(min_learning_rate, lr)
            if early_stopping > early_stop:
                break

    # Predict test data
    network.restore(os.path.join(args.logdir, "model"))
    with open("{}/lemmatizer_sota_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        forms = test.factors[test.FORMS].strings
        lemmas = network.predict(test, args.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t{}\t_".format(forms[s][i], lemmas[s][i]), file=test_file)
            print("", file=test_file)
