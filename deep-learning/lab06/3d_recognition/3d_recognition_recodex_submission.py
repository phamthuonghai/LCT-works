# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._voxels = data[\"voxels\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio, bag_id=0):
        l_voxels = len(self._voxels)
        partition_size = int(l_voxels * ratio)
        first_split = l_voxels - (bag_id+1) * partition_size
        second_split = l_voxels - bag_id * partition_size

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels = np.concatenate((self._voxels[:first_split], self._voxels[second_split:]), axis=0)
        second._voxels = self._voxels[first_split:second_split]
        if self._labels is not None:
            first._labels = np.concatenate((self._labels[:first_split], self._labels[second_split:]), axis=0)
            second._labels = self._labels[first_split:second_split]
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
                tf.float32, [None, param.modelnet_dim, param.modelnet_dim, param.modelnet_dim, 1], name=\"voxels\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")
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
                    self.loss, global_step=global_step, name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/lr\", self.learning_rate),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def train_batch(self, voxels, labels, lr):
        self.session.run([self.training, self.summaries[\"train\"]],
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


if __name__ == \"__main__\":
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
    parser.add_argument(\"--params\", default=\"params.json\", type=str, help=\"Param file path.\")
    parser.add_argument(\"--epochs\", default=300, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=16, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate\", default=0.01, type=float, help=\"Initial learning rate.\")
    parser.add_argument(\"--min_learning_rate\", default=1e-3, type=float, help=\"Minimum learning rate.\")
    parser.add_argument(\"--lr_drop_max\", default=5, type=int, help=\"Number of epochs to drop learning rate.\")
    parser.add_argument(\"--lr_drop_rate\", default=0.7, type=float, help=\"Rate of dropping learning rate.\")
    parser.add_argument(\"--early_stop\", default=20, type=int, help=\"Number of epochs to endure before early stopping.\")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        param_list = json.load(f)
        num_retry = 0
        n_params = len(param_list)
        while True:
            param = param_list[random.randint(0, n_params - 1)]
            logdir = \"logs/{}\".format(
                \",\".join(
                    \"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(param.items()))
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

    print(\"=====================================================\")
    print(param['logdir'])
    print(\"=====================================================\")

    # Load the test data
    test = Dataset(\"modelnet{}-test.npz\".format(param['modelnet_dim']), shuffle_batches=False)

    if 'bagging' in param:
        n_bags = int(1. / param['train_split'])
    else:
        n_bags = 1
    param_dict = param
    par_logdir = param_dict['logdir']

    for bag_id in range(n_bags):
        param_dict['logdir'] = os.path.join(par_logdir, str(bag_id))
        param = namedtuple('Params', param_dict.keys())(*param_dict.values())

        # Load the data
        train, dev = Dataset(\"modelnet{}-train.npz\".format(param.modelnet_dim)).split(param.train_split, bag_id)
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

            cur_loss, cur_acc = network.evaluate(\"dev\", dev, param.batch_size)
            print(\"Acc: %f, loss: %f\" % (cur_acc, cur_loss))
            sys.stdout.flush()
            if cur_loss < min_loss:
                min_loss = cur_loss
                network.save(os.path.join(param.logdir, \"model\"))
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping % args.lr_drop_max == 0:
                    lr *= args.lr_drop_rate
                    lr = max(args.min_learning_rate, lr)
                if early_stopping > args.early_stop:
                    break

        # Predict test data
        network.restore(os.path.join(param.logdir, \"model\"))
        with open(\"{}/3d_recognition_test.txt\".format(param.logdir), \"w\") as test_file:
            labels = network.predict(test, param.batch_size)
            for label in labels:
                test_file.write('%d\\n' % label)
"""

source_2 = """import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


prev_res = {}

for f in glob.glob('./results/*.txt'):
    prev_res[f] = np.loadtxt(f, dtype=int)

# Duplicate the top-2 best
prev_res['./results/6R-32-4.txt_2'] = prev_res['./results/6R-32-4.txt']
prev_res['./results/6R-32-4.txt_3'] = prev_res['./results/6R-32-4.txt']
prev_res['./results/5R-32-4.txt_2'] = prev_res['./results/5R-32-4.txt']

df = pd.DataFrame.from_dict(prev_res)

res = df.mode(axis=1)[0]

np.savetxt('./results/res.res', res.values, fmt='%d')

for col in df:
    print('Res diff to %s = %d' % (col, sum(df[col] != res)))

for the_id in range(20):
    col = './results/6R-32-4.txt'
    test = np.load('modelnet20-test.npz')
    mis_match_ids = df[col] != res
    test_sample = test['voxels'][mis_match_ids].squeeze()
    print('Ensemble: %d' % res[mis_match_ids].iloc[the_id])
    for col in df:
        print('%s: %d' % (col, df[col][mis_match_ids].iloc[the_id]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(test_sample[the_id], edgecolor='w')
    plt.show()
"""

source_3 = """import argparse
import json
import itertools

params_set = {
    'modelnet_dim': [
        20,
        # 32,
    ],
    'train_split': [
        # 0.1,
        0.2,
    ],
    'batch_size': [
        16,
        32,
        # 64,
    ],
    'model': [
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300',
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300,R-300',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500,R-500',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300,R-300,R-300',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500,R-500,R-500',
    ],
    'bagging': [
        1,
    ],
}


def is_valid(item):
    # return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']
    return True


full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

parser = argparse.ArgumentParser()
parser.add_argument(\"--params\", default=\"params.json\", type=str, help=\"Param file path.\")
args = parser.parse_args()
with open(args.params, 'w') as f:
    json.dump(full_set, f)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;0G51L|p(Ff`djUCncd$K_AVKhofrotXe2>U4{^6w-Lq@e%xJdx3vm#W3G_Wa$}_sJ!8&G5Hn1~t(mNcLXE$$zb`>s_mUo`7Au>UTAJbKMEDL1FrWj=z>_P7{tr?4n<KEfVWAWb6orJ*i@SYIN%S<uU+*UFV1nmG+`W|cA(aI3-<8ON?wwnRIvud9(CVAwDY+s3+&Wa5Ux)Z%XTQ|Yj>cA5&&x3)RHp2Op<`^}cac5qr5hOAc<RDttm$wuiWrUV-VRfh$(OWbf*5Y;J=r%1;aVJOy_{-RDn2<K0U=t!UZbwXJic_-Umb5-A+y#vx5*hTf^a&LchDb$Cr!}LC1*-}EedSODz&JeFIPMorFH*{ukR#VBWV5EE3O1bB9gnWU+Oy1@gz$8*&HF+(bTFCsY|8D=S20CqEH{yRkZ_%39e>?eTwixvfeJmJ11VibXu8<i1FfWBcE^@F<f`&G6Vx5+{bBZ5Z=2AG^N21cWa#T&X=-dyFbJmXFU4TMD#T8k^gj_$2(xHPepiy0W;gwJBx?3A_Vd{*rd)65wGn-8#S-z5kZP}?7==e`crexl#O>fihVPux*>*F$^eI-$6+RLsfB%I-S(wFt-d!G@L9PFwojJbLm6cNY+em-tvAOQTWUtrOqGGlz>hEK_2H-VC;gm|Y_Da7-~6+9f%;$6l9rf=i?U`JUWafT^?)49(Gz5nh-NI7T7*Tg3`TbzKf^h1{6hXkL`B-lQ3)Kv9Ub@pIIa$u0Y!J?00H0xm<|8{-i`ZGvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
