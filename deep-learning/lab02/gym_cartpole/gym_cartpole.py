#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Define the model, with the output layers for actions in `output_layer`
            if args.activation == 'relu':
                activation = tf.nn.relu
            elif args.activation == 'tanh':
                activation = tf.nn.tanh
            elif args.activation == 'sigmoid':
                activation = tf.nn.sigmoid
            else:
                activation = None

            hidden_layer = self.observations
            for l_id in range(args.layers):
                hidden_layer = tf.layers.dense(hidden_layer, args.hidden_layer, activation=activation,
                                               name="hidden_layer_%d" % l_id)
            output_layer = tf.layers.dense(hidden_layer, self.ACTIONS, activation=None, name="output_layer")
            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            # Global step
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(
                loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = {"train": [tf.contrib.summary.scalar("train/loss", loss),
                                            tf.contrib.summary.scalar("train/accuracy", self.accuracy)]}

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]
            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        result = self.session.run([self.accuracy, self.training, self.summaries["train"]],
                                  {self.observations: observations, self.labels: labels})
        return result[0]

    def evaluate(self, dataset, observations, labels):
        result = self.session.run([self.accuracy] + self.summaries[dataset],
                                  {self.observations: observations, self.labels: labels})
        return result[0]

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--activation", default="tanh", type=str, help="Activation function.")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Size of the hidden layer.")
    parser.add_argument("--layers", default=3, type=int, help="Number of hidden layers.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--dev_split", default=0.2, type=float, help="Dev split percentage.")
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
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    split_id = int(len(observations) * (1-args.dev_split))
    train_set = (observations[:split_id], labels[:split_id])
    dev_set = (observations[split_id:], labels[split_id:])

    # Train
    for i in range(args.epochs):
        perm_ids = np.random.permutation(len(train_set[0]))
        for _id in range(0, len(train_set[0]), args.batch_size):
            print("Epoch %d, batch %d: %f\n" % (i, _id,
                network.train(train_set[0][perm_ids[_id:_id+args.batch_size]],
                              train_set[1][perm_ids[_id:_id+args.batch_size]])))

        if len(dev_set[0]) > 0:
            print("=== Dev at epoch %d: %f\n" % (i, network.evaluate("dev", dev_set[0], dev_set[1])))

    # Save the network
    network.save("gym_cartpole/model")
