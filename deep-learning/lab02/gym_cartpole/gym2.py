import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            hidden = self.observations

            activations = {
                    "relu": tf.nn.relu,
                    "tanh": tf.nn.tanh,
                    "sigmoid": tf.nn.sigmoid,
                    "none": None
            }

            activation = activations.get(args.activation, None)

            self.training_flag = tf.placeholder_with_default(False, (), name="training_flag")

            for i in range(args.layers):
                with tf.name_scope("layer{}".format(i)):
                    hidden = tf.layers.dense(hidden, args.neurons, activation=activation)

            output_layer = tf.layers.dense(hidden, 2, activation=None)
            # TODO: Define the model, with the output layers for actions in `output_layer`

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            # Global step
            xentropy = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="xentropy")
            loss = tf.add_n([xentropy] + reg_losses, name="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = {}

                for run in ["train", "dev"]:
                    self.summaries[run] = [tf.contrib.summary.scalar("{}/loss".format(run), loss),
                                           # tf.contrib.summary.scalar("{}/xentropy".format(run), xentropy),
                                           tf.contrib.summary.scalar("{}/accuracy".format(run), self.accuracy)]

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        acc, _, _ = self.session.run([self.accuracy, self.training, self.summaries["train"]], {self.observations: observations,
                                                                    self.labels: labels,
                                                                    self.training_flag: True})
        return acc

    def evaluate(self, observations, labels):
        acc, _ = self.session.run([self.accuracy, self.summaries["dev"]], {self.observations: observations,
                                                                          self.labels: labels,
                                                                          self.training_flag: False})
        return acc

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
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--neurons", default=20, type=int, help="Number of neurons in the hidden layers.")
    parser.add_argument("--layers", default=3, type=int, help="Number of hidden layers.")
    parser.add_argument("--activation", default="tanh", type=str, help="Activation function, one of 'relu', 'tanh', 'sigmoid', 'none'")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

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

    mb_size = 5

    # Train
    for i in range(args.epochs):
        idx = np.random.permutation(len(observations))
        masks = np.array_split(idx, mb_size)

        for mask in masks[:-1]:
            acc = network.train(observations[mask], labels[mask])
            print("Acc: {:.2f}".format(acc))

        valid_acc = network.evaluate(observations[masks[-1]], labels[masks[-1]])
        if i % 20 == 0:
            print("Valid: {:.2f}".format(valid_acc))

    # Save the network
    network.save("gym_cartpole/model")