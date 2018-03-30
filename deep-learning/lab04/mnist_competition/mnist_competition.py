#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, args, batches_per_epoch=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                     intra_op_parallelism_threads=args.threads))
        with self.session.graph.as_default():
            # Construct the network and training operation.
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            self.predictions = None
            self.loss = None
            self.build()

            # Training
            global_step = tf.train.create_global_step()
            if args.learning_rate_final is not None and batches_per_epoch is not None:
                lr = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch,
                                                (args.learning_rate_final/args.learning_rate) ** (1.0/(args.epochs-1)),
                                                staircase=True)
            else:
                lr = args.learning_rate
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                    self.loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def build(self):
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
            elif def_params[0] == 'RB':
                features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
                features = tf.layers.batch_normalization(features, training=self.is_training)
                features = tf.nn.relu(features)
            elif def_params[0] == 'CB':
                features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                            strides=int(def_params[3]), padding=def_params[4], activation=None,
                                            use_bias=False)
                features = tf.layers.batch_normalization(features, training=self.is_training)
                features = tf.nn.relu(features)
            else:
                continue

        output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
        self.predictions = tf.argmax(output_layer, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
                                       {self.images: images, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, dataset, images, labels):
        labels = np.clip(labels, 0, 9)
        preds, _ = self.session.run([self.predictions, self.summaries[dataset]],
                                    {self.images: images, self.labels: labels, self.is_training: False})
        return preds

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=None, type=str, help="Params file.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
    parser.add_argument("--cnn", default="CB-10-3-2-same,M-3-2,F,R-100", type=str,
                        help="Description of the CNN architecture.")
    args = parser.parse_args()

    if args.params is not None:
        import json
        import random
        from collections import namedtuple
        with open(args.params, 'r') as f:
            param_list = json.load(f)
            while True:
                param = param_list[random.randint(0, len(param_list)-1)]
                logdir = "logs/{}-{}".format(
                    os.path.basename(__file__),
                    ",".join(
                        "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(param.items()))
                )
                if not os.path.exists(logdir):
                    param['logdir'] = logdir
                    args = namedtuple('Params', param.keys())(*param.values())
                    break
    else:
        # Create logdir name
        args.logdir = "logs/{}-{}".format(
            os.path.basename(__file__),
            ",".join(("{}={}".format(
                re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
        )

    if not os.path.exists("logs"):
        os.mkdir("logs")  # TF 1.6 will do this by itself

    print("=====================================================")
    print(args.logdir)
    print("=====================================================")

    # Load the data
    from tensorflow.examples.tutorials import mnist

    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42)

    # Construct the network
    batches_per_epoch = mnist.train.num_examples // args.batch_size
    network = Network(args, batches_per_epoch)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        dev_acc = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        print("Dev: {:.2f}".format(100 * dev_acc))

    network.save(os.path.join(args.logdir, "model"))

    # Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predicts = network.predict("test", mnist.test.images, mnist.test.labels)
    with open(os.path.join(args.logdir, "predict.txt"), "w") as f:
        f.write('\n'.join(predicts.astype(str))+'\n')
