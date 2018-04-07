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
            self.learning_rate = tf.placeholder_with_default(0.01, None)

            # Computation
            loss, masks_loss, self.labels_predictions, self.masks_predictions = self.build_model(params)
            self.loss = loss + params.masks_loss_coef * masks_loss

            # Training
            global_step = tf.train.create_global_step()

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=global_step, name="training")

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
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/loss_cls", loss),
                                           tf.contrib.summary.scalar("train/loss_masks", masks_loss),
                                           tf.contrib.summary.scalar("train/lr", self.learning_rate),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", self.iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.loss),
                                               tf.contrib.summary.scalar(dataset + "/loss_cls", loss),
                                               tf.contrib.summary.scalar(dataset + "/loss_masks", masks_loss),
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

    def train(self, images, labels, masks, lr):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks,
                          self.is_training: True, self.learning_rate: lr})

    def evaluate(self, dataset, images, labels, masks):
        loss, iou, _ = self.session.run([self.loss, self.iou, self.summaries[dataset]],
                                        {self.images: images, self.labels: labels,
                                         self.masks: masks, self.is_training: False})
        return loss, iou

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})

    def save(self, path):
        self.saver.save(self.session, path)

    def restore(self, path):
        self.saver.restore(self.session, path)


def get_layer(layer_def, features, is_training):
    def_params = layer_def.split('-')
    if def_params[0] == 'C':
        features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                    strides=int(def_params[3]), padding='same', activation=tf.nn.relu)
    elif def_params[0] == 'M':
        features = tf.layers.max_pooling2d(features, pool_size=int(def_params[1]),
                                           strides=int(def_params[2]), padding='same')
    elif def_params[0] == 'F':
        features = tf.layers.flatten(features)
    elif def_params[0] == 'R':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=tf.nn.relu)
    elif def_params[0] == 'RB':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'RD':
        features = tf.layers.dense(features, units=int(def_params[1]), activation=None, use_bias=False)
        features = tf.layers.dropout(features, rate=float(def_params[2]), training=is_training)
    elif def_params[0] == 'CB':
        features = tf.layers.conv2d(features, filters=int(def_params[1]), kernel_size=int(def_params[2]),
                                    strides=int(def_params[3]), padding='same', activation=None,
                                    use_bias=False)
        features = tf.layers.batch_normalization(features, training=is_training)
        features = tf.nn.relu(features)
    elif def_params[0] == 'D':
        features = tf.layers.dropout(features, rate=float(def_params[1]), training=is_training)
    return features


class CNN(Network):
    def build_model(self, params):
        features = self.images

        part_defs = params.cnn.strip().split(';')

        # Common part
        common_layer = features
        for layer_def in part_defs[0].split(','):
            common_layer = get_layer(layer_def, common_layer, self.is_training)

        # Classification part
        classify_layer = common_layer
        for layer_def in part_defs[1].split(','):
            classify_layer = get_layer(layer_def, classify_layer, self.is_training)

        labels_output_layer = tf.layers.dense(classify_layer, self.LABELS, activation=None)
        labels_predictions = tf.argmax(labels_output_layer, axis=1)

        loss = tf.losses.sparse_softmax_cross_entropy(self.labels, labels_output_layer)

        # Masking part
        mask_layer = common_layer
        for layer_def in part_defs[2].split(','):
            mask_layer = get_layer(layer_def, mask_layer, self.is_training)

        masks_predictions = tf.round(tf.sigmoid(mask_layer))
        masks_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.masks, logits=mask_layer))

        return loss, masks_loss, labels_predictions, masks_predictions


def get_model(name):
    return getattr(sys.modules[__name__], name)
