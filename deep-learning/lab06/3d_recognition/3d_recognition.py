#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._voxels = data["voxels"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


def get_layer(layer_def, features, is_training):
    def_params = layer_def.split('-')
    if def_params[0] == 'C':
        features = tf.layers.conv3d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                    strides=int(def_params[3]), padding='same', activation=tf.nn.relu)
    elif def_params[0] == 'M':
        features = tf.layers.max_pooling3d(features, pool_size=int(def_params[1]),
                                           strides=int(def_params[2]), padding='same')
    elif def_params[0] == 'F':
        features = tf.layers.flatten(features)
    elif def_params[0] == 'R':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
    elif def_params[0] == 'RB':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'CB':
        features = tf.layers.conv3d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                    strides=int(def_params[3]), padding='same', activation=None,
                                    use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'D':
        features = tf.layers.dropout(features, rate=float(def_params[1]), training=is_training)
    return features


class Network:
    LABELS = 10

    def __init__(self, params, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=param.threads,
                                                                     intra_op_parallelism_threads=param.threads))

        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, param.modelnet_dim, param.modelnet_dim, param.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            # Computation and training.
            features = self.voxels

            part_defs = params.model.strip().split(';')
            for layer_def in part_defs[0].split(','):
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
            summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
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

    def train_batch(self, voxels, labels, lr):
        self.session.run([self.training, self.summaries["train"]],
                         {self.voxels: voxels, self.labels: labels, self.is_training: True, self.learning_rate: lr})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_voxels, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy],
                {self.voxels: batch_voxels, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_voxels) / len(dataset.voxels)
            accuracy += batch_accuracy * len(batch_voxels) / len(dataset.voxels)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return loss, accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            voxels, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False}))
        return np.concatenate(labels)

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import sys
    import os
    import re
    import json
    import random
    from collections import namedtuple

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json", type=str, help="Param file path.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--min_learning_rate", default=1e-3, type=float, help="Minimum learning rate.")
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
                param = namedtuple('Params', param.keys())(*param.values())
                break
            num_retry += 1
            if num_retry > n_params:
                exit(111)

    if not os.path.exists("logs"):
        os.mkdir("logs")  # TF 1.6 will do this by itself

    print("=====================================================")
    print(param.logdir)
    print("=====================================================")

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(param.modelnet_dim)).split(param.train_split)
    test = Dataset("modelnet{}-test.npz".format(param.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(param)

    # Train
    min_loss = 10000
    early_stopping = 0
    lr = args.learning_rate
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(param.batch_size)
            network.train_batch(voxels, labels, lr)

        cur_loss, cur_acc = network.evaluate("dev", dev, param.batch_size)
        print("Acc: %f, loss: %f" % (cur_acc, cur_loss))
        sys.stdout.flush()
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
    with open("{}/3d_recognition_test.txt".format(param.logdir), "w") as test_file:
        labels = network.predict(test, param.batch_size)
        for label in labels:
            test_file.write('%d\n' % label)
