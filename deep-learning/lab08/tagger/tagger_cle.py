#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_tags):
        with self.session.graph.as_default():
            if args.recodex:
                tf.get_variable_scope().set_initializer(tf.glorot_uniform_initializer(seed=42))

            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            # Choose RNN cell class according to args.rnn_cell (LSTM and GRU
            # should be supported, using tf.nn.rnn_cell.{BasicLSTM,GRU}Cell).
            rnn_cell_types = {
                'LSTM': tf.nn.rnn_cell.BasicLSTMCell,
                'GRU': tf.nn.rnn_cell.GRUCell,
            }
            fw_rnn_cell = rnn_cell_types[args.rnn_cell](args.rnn_cell_dim)
            bw_rnn_cell = rnn_cell_types[args.rnn_cell](args.rnn_cell_dim)

            # Create word embeddings for num_words of dimensionality args.we_dim
            # using `tf.get_variable`.
            word_embeddings = tf.get_variable('word_embeddings', [num_words, args.we_dim], dtype=tf.float32)

            # Embed self.word_ids according to the word embeddings, by utilizing
            # `tf.nn.embedding_lookup`.
            word_embedded = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            # Character-level word embeddings (CLE)

            # Generate character embeddings for num_chars of dimensionality args.cle_dim.
            char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim], dtype=tf.float32)

            # Embed self.charseqs (list of unique words in the batch) using the character embeddings.
            char_embedded = tf.nn.embedding_lookup(char_embeddings, self.charseqs)

            # Use `tf.nn.bidirectional_dynamic_rnn` to process embedded self.charseqs using
            # a GRU cell of dimensionality `args.cle_dim`.
            _, char_rnn_final_outputs = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim),
                tf.nn.rnn_cell.GRUCell(args.cle_dim),
                char_embedded,
                sequence_length=self.charseq_lens,
                dtype=tf.float32,
            )

            # Sum the resulting fwd and bwd state to generate character-level word embedding (CLE)
            # of unique words in the batch.
            char_rnn = char_rnn_final_outputs[0] + char_rnn_final_outputs[1]

            # Generate CLEs of all words in the batch by indexing the just computed embeddings
            # by self.charseq_ids (using tf.nn.embedding_lookup).
            cle = tf.nn.embedding_lookup(char_rnn, self.charseq_ids)

            # Concatenate the word embeddings (computed above) and the CLE (in this order).
            embedded = tf.concat([word_embedded, cle], axis=2)

            # Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction) and self.sentence_lens.
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, embedded, dtype=tf.float32,
                                                             sequence_length=self.sentence_lens)

            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            rnn_output = tf.concat(rnn_outputs, axis=2)

            # Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            output_layer = tf.layers.dense(rnn_output, num_tags)

            # Generate `self.predictions`.
            self.predictions = tf.argmax(output_layer, axis=2)

            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Training

            # Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions,
                                                                              weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size,
                                                                                            including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                              including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
    parser.add_argument("--cle_dim", default=32, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="ReCodEx mode.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-cac-train.txt", max_sentences=5000)
    dev = morpho_dataset.MorphoDataset("czech-cac-dev.txt", train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        accuracy = network.evaluate("dev", dev, args.batch_size)
        print("{:.2f}".format(100 * accuracy))
