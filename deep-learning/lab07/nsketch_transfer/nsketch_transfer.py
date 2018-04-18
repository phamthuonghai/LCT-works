#!/usr/bin/env python3

# This source depends on the NASNet A Mobile network, which can be downloaded
# from http://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/nasnet_a_mobile.zip.

import numpy as np
import tensorflow as tf

try:
    from nets.nasnet import nasnet
except Exception as e:
    from .nets.nasnet import nasnet


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images))\
            if self._shuffle_batches else np.arange(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images))\
                if self._shuffle_batches else np.arange(len(self._images))
            return True
        return False


def get_layer(layer_def, features, is_training):
    def_params = layer_def.split('-')
    if def_params[0] == 'F':
        features = tf.layers.flatten(features)
    elif def_params[0] == 'R':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
    elif def_params[0] == 'RB':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'D':
        features = tf.layers.dropout(features, rate=float(def_params[1]), training=is_training)
    return features


class Network:
    WIDTH, HEIGHT = 224, 224
    LABELS = 250
    CHECKPOINTS = {
        'nasnet': 'nets/nasnet/model.ckpt'
    }

    def __init__(self, args, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                     intra_op_parallelism_threads=args.threads))

        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
                features, _ = nasnet.build_nasnet_mobile(images, num_classes=None, is_training=args.train_again)
            self.nasnet_saver = tf.train.Saver()

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`
            for layer_def in args.model.strip().split(';'):
                features = get_layer(layer_def, features, self.is_training)
            output = tf.layers.dense(features, self.LABELS, activation=None)
            self.predictions = tf.argmax(output, axis=1)
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output)

            global_step = tf.train.create_global_step()
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/lr", self.learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

            # Load NASNet
            self.nasnet_saver.restore(self.session, self.CHECKPOINTS[args.pretrained])

    def train_batch(self, images, labels, learning_rate):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels,
                          self.is_training: True, self.learning_rate: learning_rate})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy],
                {self.images: batch_images, self.labels: batch_labels, self.is_training: False})

            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy, loss

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import json
    import os
    import re
    import sys
    import random
    from collections import namedtuple

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json", type=str, help="Param file path.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--min_learning_rate", default=1e-4, type=float, help="Minimum learning rate.")
    parser.add_argument("--lr_drop_max", default=5, type=int, help="Number of epochs to drop learning rate.")
    parser.add_argument("--lr_drop_rate", default=0.7, type=float, help="Rate of dropping learning rate.")
    parser.add_argument("--early_stop", default=20, type=int, help="Number of epochs to endure before early stopping.")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        param_list = json.load(f)
        num_retry = 0
        n_params = len(param_list)
        while True:
            param = param_list[random.randint(0, n_params - 1)]
            logdir = "logs/{}".format(
                ",".join(
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(param.items()))
            )
            if not os.path.exists(logdir):
                param['logdir'] = logdir
                param['epochs'] = args.epochs
                param['threads'] = args.threads
                break
            num_retry += 1
            if num_retry > n_params:
                exit(111)

    os.makedirs(param['logdir'])

    print("=====================================================")
    print(param['logdir'])
    print("=====================================================")

    param = namedtuple('Params', param.keys())(*param.values())

    # Load the data
    train = Dataset("./data/nsketch-train.npz")
    dev = Dataset("./data/nsketch-dev.npz", shuffle_batches=False)
    test = Dataset("./data/nsketch-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network(param)

    # Train
    min_loss = 10000
    early_stopping = 0
    recent_losses = []
    lr = args.learning_rate
    for i in range(args.epochs):
        while not train.epoch_finished():
            images, labels = train.next_batch(param.batch_size)
            network.train_batch(images, labels, lr)

        cur_acc, cur_loss = network.evaluate("dev", dev, param.batch_size)

        print("Acc: %f, loss: %f" % (cur_acc, cur_loss))
        sys.stdout.flush()

        # To avoid spikes
        recent_losses.append(cur_loss)
        if len(recent_losses) > 5:
            recent_losses = recent_losses[1:]
        cur_loss = sum(recent_losses) / len(recent_losses)

        if cur_loss < min_loss:
            min_loss = cur_loss
            network.save(os.path.join(param.logdir, "model"))
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping % args.lr_drop_max == 0:
                lr *= args.lr_drop_rate
                lr = max(args.min_learning_rate, lr)
            if early_stopping > args.early_stop:
                break

    # Predict test data
    network.restore(os.path.join(param.logdir, "model"))
    with open("{}/nsketch_transfer_test.txt".format(param.logdir), "w") as test_file:
        labels = network.predict(test, param.batch_size)
        for label in labels:
            print(label, file=test_file)
