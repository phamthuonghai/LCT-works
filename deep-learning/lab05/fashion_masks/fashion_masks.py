#!/usr/bin/env python3
import numpy as np
import sys

try:
    from .models import get_model
except Exception:
    from models import get_model


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(
            len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    @property
    def num_examples(self):
        return len(self._images)

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None, self._masks[
            batch_perm] if self._masks is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else range(
                len(self._images))
            return True
        return False


def main():
    import argparse
    import os
    import re
    import json
    import random
    from collections import namedtuple

    # Load the data
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz", shuffle_batches=False)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params_cnn.json", type=str, help="Param file path.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--min_learning_rate", default=1e-3, type=float, help="Minimum learning rate.")
    parser.add_argument("--lr_drop_max", default=5, type=int, help="Number of epochs to drop learning rate.")
    parser.add_argument("--lr_drop_rate", default=0.7, type=float, help="Rate of dropping learning rate.")
    parser.add_argument("--early_stop", default=20, type=int, help="Number of epochs to endure before early stopping.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    _args = parser.parse_args()

    with open(_args.params, 'r') as f:
        param_list = json.load(f)
        num_retry = 0
        n_params = len(param_list)
        while True:
            param = param_list[random.randint(0, n_params - 1)]
            logdir = "logs/{}-{}".format(
                os.path.basename(__file__),
                ",".join(
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(param.items()))
            )
            if not os.path.exists(logdir):
                param['logdir'] = logdir
                param['epochs'] = _args.epochs
                param['threads'] = _args.threads
                param['batches_per_epoch'] = train.num_examples // param['batch_size']
                param = namedtuple('Params', param.keys())(*param.values())
                break
            num_retry += 1
            if num_retry > n_params:
                exit(1)

    if not os.path.exists("logs"):
        os.mkdir("logs")  # TF 1.6 will do this by itself

    print("=====================================================")
    print(param.logdir)
    print("=====================================================")

    # Construct the network
    network = get_model(param.model)(param)

    # Train
    min_loss = 10000
    early_stopping = 0
    lr = _args.learning_rate
    for i in range(_args.epochs):
        while not train.epoch_finished():
            images, labels, masks = train.next_batch(param.batch_size)
            network.train(images, labels, masks, lr)

        cur_loss, cur_iou = network.evaluate("dev", dev.images, dev.labels, dev.masks)
        print("IoU: %f, loss: %f" % (cur_iou, cur_loss))
        sys.stdout.flush()
        if cur_loss < min_loss:
            min_loss = cur_loss
            network.save(os.path.join(param.logdir, "model"))
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping % _args.lr_drop_max == 0:
                lr *= _args.lr_drop_rate
                lr = max(_args.min_learning_rate, lr)
            if early_stopping > _args.early_stop:
                break

    # Predict test data
    network.restore(os.path.join(param.logdir, "model"))
    with open(os.path.join(param.logdir, "fashion_masks_test.txt"), "w") as test_file:
        while not test.epoch_finished():
            images, _, _ = test.next_batch(param.batch_size)
            labels, masks = network.predict(images)
            for i in range(len(labels)):
                print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)


if __name__ == "__main__":
    main()
