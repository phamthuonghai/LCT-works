#!/usr/bin/env python3
import morpho_dataset
from models import get_model


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for col_id in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[col_id], columns[col_id + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


if __name__ == "__main__":
    import argparse
    import json
    import sys
    import os
    import re
    import random
    from collections import namedtuple

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.json", type=str, help="Param file path.")
    parser.add_argument("--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        param_list = json.load(f)
        num_retry = 0
        n_params = len(param_list)
        while True:
            param = param_list[random.randint(0, n_params - 1)]
            flat_param = {}
            for k, v in param.items():
                if isinstance(v, dict):
                    flat_param.update(v)
                else:
                    flat_param[k] = v
            param = flat_param
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

    os.makedirs(param.logdir)

    print("=====================================================")
    print(param.logdir)
    print("=====================================================")
    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = get_model(param.name)(param,
                                    len(train.factors[train.FORMS].words),
                                    len(train.factors[train.FORMS].alphabet),
                                    len(train.factors[train.TAGS].words))

    # Train
    min_loss = 10000
    early_stopping = 0
    lr = param.learning_rate
    for e_id in range(param.epochs):
        network.train_epoch(train, param.batch_size, lr)

        cur_acc, cur_loss = network.evaluate("dev", dev, param.batch_size)
        print("#%3d: acc: %f, loss: %f" % (e_id, cur_acc, cur_loss))
        sys.stdout.flush()
        if cur_loss < min_loss:
            min_loss = cur_loss
            network.save(os.path.join(param.logdir, "model"))
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping % param.lr_drop_max == 0:
                lr *= param.lr_drop_rate
                lr = max(param.min_learning_rate, lr)
            if early_stopping > param.early_stop:
                break

    # Predict test data
    network.restore(os.path.join(param.logdir, "model"))
    with open("{}/tagger_sota_test.txt".format(param.logdir), "w", encoding='utf8') as test_file:
        forms = test.factors[test.FORMS].strings
        tags = network.predict(test, param.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t_\t{}".format(forms[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
            print("", file=test_file)
