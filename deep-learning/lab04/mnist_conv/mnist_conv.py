#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            # Add layers described in the args.cnn. Layers are separated by a comma and can be:
            # - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
            #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
            # - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
            # - F: Flatten inputs
            # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
            # Store result in `features`.
            features = self.images
            layer_defs = args.cnn.strip().split(',')
            for layer_def in layer_defs:
                def_params = layer_def.split('-')
                if def_params[0] == 'C':
                    features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                                strides=int(def_params[3]), padding=def_params[4],
                                                activation=tf.nn.relu)
                elif def_params[0] == 'M':
                    features = tf.layers.max_pooling2d(features, pool_size=int(def_params[1]),
                                                       strides=int(def_params[2]))
                elif def_params[0] == 'F':
                    features = tf.layers.flatten(features)
                elif def_params[0] == 'R':
                    features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
                else:
                    continue

            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

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

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels})
        return accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        dev_acc = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        # print("Dev: {:.2f}".format(100 * dev_acc))

    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)
    print("{:.2f}".format(100 * accuracy))
