import sys

import tensorflow as tf


class Model:
    def __init__(self, args, num_words, num_chars, num_tags):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                     intra_op_parallelism_threads=args.threads))

        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            output_layer, self.predictions = self.build_model(args, num_words, num_chars, num_tags)

            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Training

            # Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights)

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions,
                                                                              weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/lr", self.learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def build_model(self, args, num_words, num_chars, num_tags):
        raise NotImplemented

    def train_epoch(self, train, batch_size, lr):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size,
                                                                                            including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries["train"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS],
                              self.learning_rate: lr,
                              self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                              including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[dataset.FORMS], self.charseq_lens: charseq_lens[dataset.FORMS],
                              self.word_ids: word_ids[dataset.FORMS], self.charseq_ids: charseq_ids[dataset.FORMS],
                              self.tags: word_ids[dataset.TAGS],
                              self.is_training: False})
        acc, loss, _ = self.session.run([self.current_accuracy, self.current_loss, self.summaries[dataset_name]])
        return acc, loss

    def predict(self, dataset, batch_size):
        tags = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                              including_charseqs=True)
            tags.extend(self.session.run(self.predictions,
                                         {self.sentence_lens: sentence_lens,
                                          self.charseqs: charseqs[dataset.FORMS],
                                          self.charseq_lens: charseq_lens[dataset.FORMS],
                                          self.word_ids: word_ids[dataset.FORMS],
                                          self.charseq_ids: charseq_ids[dataset.FORMS],
                                          self.is_training: False}))
        return tags

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)


class CLE(Model):
    def build_model(self, args, num_words, num_chars, num_tags):
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
        word_embeddings = tf.layers.dropout(word_embeddings, rate=args.we_dropout,
                                            noise_shape=[num_words, 1], training=self.is_training)

        # Embed self.word_ids according to the word embeddings, by utilizing
        # `tf.nn.embedding_lookup`.
        word_embedded = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

        # Character-level word embeddings (CLE)

        # Generate character embeddings for num_chars of dimensionality args.cle_dim.
        char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim], dtype=tf.float32)
        # char_embeddings dropout
        char_embeddings = tf.layers.dropout(char_embeddings, rate=args.cle_dropout,
                                            noise_shape=[num_chars, 1], training=self.is_training)

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
            scope='char_rnn'
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
                                                         sequence_length=self.sentence_lens, scope='mixed_rnn')

        # Concatenate the outputs for fwd and bwd directions (in the third dimension).
        rnn_output = tf.concat(rnn_outputs, axis=2)

        rnn_output = tf.layers.dropout(rnn_output, rate=args.dropout, training=self.is_training)
        # Add a dense layer (without activation) into num_tags classes and
        # store result in `output_layer`.
        output_layer = tf.layers.dense(rnn_output, num_tags)

        # Generate `predictions`.
        predictions = tf.argmax(output_layer, axis=2)

        return output_layer, predictions


def get_model(name):
    return getattr(sys.modules[__name__], name)
