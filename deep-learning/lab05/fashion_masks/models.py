import sys

import tensorflow as tf


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, params):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=params.threads,
                                                                     intra_op_parallelism_threads=params.threads))

        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            loss, masks_loss, self.labels_predictions, self.masks_predictions = self.build_model(params)

            # Training
            global_step = tf.train.create_global_step()
            if params.learning_rate_final is not None and params.batches_per_epoch is not None:
                lr = tf.train.exponential_decay(params.learning_rate, global_step, params.batches_per_epoch,
                                                (params.learning_rate_final/params.learning_rate) **
                                                (1.0/(params.epochs-1)),
                                                staircase=True)
            else:
                lr = params.learning_rate
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=lr).minimize(
                    params.train_loss_coef * loss + (1 - params.train_loss_coef) * masks_loss,
                    global_step=global_step, name="training")

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1, 2, 3])
            self.iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1, 2, 3]) +
                                tf.reduce_sum(self.masks, axis=[1, 2, 3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/masks_loss", masks_loss),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/masks_loss", masks_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", accuracy),
                                               tf.contrib.summary.scalar(dataset + "/iou", self.iou),
                                               tf.contrib.summary.image(dataset + "/images", self.images),
                                               tf.contrib.summary.image(dataset + "/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            self.saver = tf.train.Saver()

    def build_model(self, params):
        """
        :param params:
        :return:
            loss:
            masks_loss:
            labels_predictions: shape [None] and type tf.int64
            masks_predictions: shape [None, 28, 28, 1] and type tf.float32, values 0 or 1
        """
        raise NotImplementedError()

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        iou, _ = self.session.run([self.iou, self.summaries[dataset]],
                                  {self.images: images, self.labels: labels,
                                   self.masks: masks, self.is_training: False})
        return iou

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)


class CNN(Network):
    def build_model(self, params):
        features = self.images
        layer_defs = params.cnn.strip().split(',')
        masks_output_layer = None
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
            elif def_params[0] == 'O':
                masks_output_layer = features
            else:
                continue

        labels_output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="labels_output_layer")
        labels_predictions = tf.argmax(labels_output_layer, axis=1)
        masks_predictions = tf.round(tf.sigmoid(masks_output_layer))
        loss = tf.losses.sparse_softmax_cross_entropy(self.labels, labels_output_layer, scope="loss")
        masks_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.masks, logits=masks_output_layer)
        return loss, masks_loss, labels_predictions, masks_predictions


def get_model(name):
    return getattr(sys.modules[__name__], name)
