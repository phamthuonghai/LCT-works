#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        tf.logging.info("Loading data from " + filename)

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet:
                    break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.int32)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map:
                char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = int(self._text[i].isupper())

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False

    def text_with_mask(self, mask):
        return ''.join([ch if mask[i] == 0 else ch.upper() for i, ch in enumerate(self._text)])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.int32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")

            # Define a suitable network with appropriate loss function
            char_embeddings = tf.get_variable("char_embeddings", [args.alphabet_size, args.embedding_size])
            embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.windows)
            embedded_chars = tf.layers.flatten(embedded_chars)

            activation = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid }
            hidden_layer = embedded_chars
            for layer_id in range(args.layers):
                hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer_units,
                                               activation=activation.get(args.activation, None),
                                               name="hidden_layer_%d" % layer_id)

            output = tf.layers.dense(hidden_layer, 2, activation=None, name="output")
            self.predictions = tf.argmax(output, axis=1, output_type=tf.int32, name="predictions")

            # Define training
            loss = tf.losses.sparse_softmax_cross_entropy(logits=output, labels=self.labels)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step)

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels})

    def evaluate(self, dataset, windows, labels):
        acc, _ = self.session.run([self.accuracy, self.summaries[dataset]],
                                  {self.windows: windows, self.labels: labels})
        return acc

    def predict(self, dataset, windows, labels):
        preds, _ = self.session.run([self.predictions, self.summaries[dataset]],
                                    {self.windows: windows, self.labels: labels})
        return preds

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=100, type=int, help="Alphabet size.")
    parser.add_argument("--embedding_size", default=10, type=int, help="Embedding size.")
    parser.add_argument("--batch_size", default=4000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=80, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=5, type=int, help="Size of the window to use.")
    parser.add_argument("--test", dest='test', action='store_true', help="Should output test file.")
    parser.add_argument("--hidden_layer_units", default=20, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=3, type=int, help="Number of layers.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--activation", default="relu", type=str, help="Activation function.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)

    with open(os.path.join(args.logdir, "vocab.txt"), "w") as f:
        f.write("\n".join([repr(c) for c in train.alphabet]))

    # Construct the network
    tf.logging.info("Constructing the network")
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    tf.logging.info("Training")
    dev_windows, dev_labels = dev.all_data()
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)

        dev_acc = network.evaluate("dev", dev_windows, dev_labels)
        tf.logging.info("{:.2f}".format(100 * dev_acc))

    network.save(os.path.join(args.logdir, "model"))

    # Generate the uppercased test set
    if args.test:
        test_windows, test_labels = test.all_data()
        mask = network.predict("test", test_windows, test_labels)
        with open(os.path.join(args.logdir, "predict.txt"), "w") as f:
            f.write(test.text_with_mask(mask))
