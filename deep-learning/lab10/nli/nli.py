import logging
import os
import pickle

import numpy as np
from scipy.sparse import hstack
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
n_train = 9900

default_param = True
clf = LinearSVC

# features = [('text', 'char', 2, 9, 60000), ('tags', 'word', 1, 4, 400)]
# features = [('text', 'char', 2, 9, 60000)]
# features = [('text', 'char', 2, 10, 200000), ('tags', 'word', 1, 6, 400)]
# features = [('text', 'char', 2, 10, 200000), ('tags', 'char', 1, 6, 400)]
# features = [('text', 'char', 2, 15, 200000)]
features = [('text', 'char', 2, 10, 200000)]

feature_file = '_'.join('-'.join(str(i) for i in ft) for ft in features)
output_dir = clf.__name__ + '_' + feature_file

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


def prediction_results(expected, predicted):
    print("\nConfusion Matrix:\n")
    cm = metrics.confusion_matrix(expected, predicted).tolist()
    row_format = "{:>5}" * (len(CLASSES) + 1)
    print(row_format.format("", *CLASSES))
    for l1, row in zip(CLASSES, cm):
        print(row_format.format(l1, *row))
    print("\nClassification Results:\n")
    print(metrics.classification_report(expected, predicted, target_names=CLASSES))


def output_results(predicted, f_name):
    with open(f_name, 'w') as f:
        for l in predicted:
            f.write(CLASSES[l] + '\n')


def data_reader(f_name, label_only=False):
    logging.info('Loading data from %s' % f_name)
    data = {'target': [], 'text': [], 'tags': []}
    with open(f_name, encoding='utf-8') as f:
        for line in f:
            parsed_line = line.rstrip().split('\t')
            if 'test' not in f_name:
                data['target'].append(CLASSES.index(parsed_line[0]))
            if not label_only:
                text = []
                tags = []
                for _id, w in enumerate(parsed_line[3:]):
                    parsed_w = w.strip().split()
                    if len(parsed_w) != 2:
                        continue
                    text.append(parsed_w[0])
                    tags.append(parsed_w[1])
                data['text'].append(' '.join(text))
                data['tags'].append(' '.join(tags))
    return data


if __name__ == '__main__':
    logging.info(output_dir)
    if os.path.exists(feature_file):
        logging.info('Loading features from %s' % feature_file)
        with open(feature_file, 'rb') as ff:
            train_x, test_x, train_y = pickle.load(ff)
    else:
        train = data_reader("nli-train.txt")
        dev = data_reader("nli-dev.txt")
        test = data_reader("nli-test.txt")

        logging.info('Extracting features')
        train_x = None
        test_x = None
        for feature in features:
            logging.info(feature)
            vectorizer = TfidfVectorizer(max_features=feature[4], ngram_range=(feature[2], feature[3]),
                                         sublinear_tf=True, analyzer=feature[1])

            vectorizer.fit(train[feature[0]] + dev[feature[0]] + test[feature[0]])

            if train_x is None:
                train_x = vectorizer.transform(train[feature[0]] + dev[feature[0]])
            else:
                train_x = hstack([train_x, vectorizer.transform(train[feature[0]] + dev[feature[0]])])

            if test_x is None:
                test_x = vectorizer.transform(test[feature[0]])
            else:
                test_x = hstack([test_x, vectorizer.transform(test[feature[0]])])

        train_x = train_x.tocsr()
        test_x = test_x.tocsr()
        train_y = np.array(train['target'] + dev['target'])

        with open(feature_file, 'wb') as ff:
            pickle.dump([train_x, test_x, train_y], ff)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info('Training')

    if default_param:
        model = clf(verbose=1)
    else:
        param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100]
            },
            'SVC': [
                {'C': [1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ],
            'LinearSVC': {
                'C': [1, 10, 100],
            }
        }

        model = GridSearchCV(clf(), param_grid=param_grids[clf.__name__], verbose=1, n_jobs=32)

    model.fit(train_x[:n_train], train_y[:n_train])
    if not default_param:
        logging.info(model.best_params_)

    dev_prev = model.predict(train_x[n_train:])
    output_results(dev_prev, os.path.join(output_dir, 'dev.txt'))
    prediction_results(train_y[n_train:], dev_prev)

    logging.info('Predicting')
    test_prev = model.predict(test_x)
    output_results(test_prev, os.path.join(output_dir, 'test.txt'))

    logging.info('Done %s' % output_dir)
