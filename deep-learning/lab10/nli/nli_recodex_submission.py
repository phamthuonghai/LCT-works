# coding=utf-8

source_1 = """import logging
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
clf = SVC

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
    print(\"\\nConfusion Matrix:\\n\")
    cm = metrics.confusion_matrix(expected, predicted).tolist()
    row_format = \"{:>5}\" * (len(CLASSES) + 1)
    print(row_format.format(\"\", *CLASSES))
    for l1, row in zip(CLASSES, cm):
        print(row_format.format(l1, *row))
    print(\"\\nClassification Results:\\n\")
    print(metrics.classification_report(expected, predicted, target_names=CLASSES))


def output_results(predicted, f_name):
    with open(f_name, 'w') as f:
        for l in predicted:
            f.write(CLASSES[l] + '\\n')


def data_reader(f_name, label_only=False):
    logging.info('Loading data from %s' % f_name)
    data = {'target': [], 'text': [], 'tags': []}
    with open(f_name, encoding='utf-8') as f:
        for line in f:
            parsed_line = line.rstrip().split('\\t')
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
        train = data_reader(\"nli-train.txt\")
        dev = data_reader(\"nli-dev.txt\")
        test = data_reader(\"nli-test.txt\")

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
"""

source_2 = """import glob


def compare(system, gold):
    if len(system) < len(gold):
        raise RuntimeError(\"The system output is shorter than gold data: {} vs {}.\".format(len(system), len(gold)))

    correct = sum([gold[i] == system[i] for i in range(len(gold))])
    return correct * 100.0 / len(gold)


def combine(cans, d):
    to_count = [[] for _ in range(len(cans[0][d]))]
    for c in cans:
        for _id, row in enumerate(c[d]):
            to_count[_id].append(row)

    return [max(row, key=row.count) for row in to_count]


if __name__ == '__main__':
    with open('nli-dev.txt') as f:
        gold_dev = [t.strip().split('\\t')[0] for t in f]

    candidates = []
    for dev_file in glob.glob('*/dev.txt'):
        with open(dev_file) as f:
            dev = [t.strip().split('\\t')[0] for t in f]
        model = dev_file.split('/')[0]
        test_file = model + '/test.txt'
        with open(test_file) as f:
            test = [t.strip().split('\\t')[0] for t in f]

        candidates.append({'model': model, 'dev': dev, 'test': test, 'dev_score': compare(dev, gold_dev)})

    candidates = sorted(candidates, key=lambda t: t['dev_score'], reverse=True)

    for can in candidates:
        print('%.5f -- %s' % (can['dev_score'], can['model']))

    print('\\nEnsembling\\n')

    max_dev_score = candidates[0]['dev_score']
    max_r_can = 1
    for r_can in range(len(candidates)):
        score = compare(combine(candidates[:r_can+1], 'dev'), gold_dev)
        print('%d -- %.5f' % (r_can, score))
        if score > max_dev_score:
            max_dev_score = score
            max_r_can = r_can

    print('Combining test output until #%d candidates' % max_r_can)

    with open('test.txt', 'w') as f:
        test = combine(candidates, 'test')
        f.write('\\n'.join(test) + '\\n')
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;1MqZ{9OPg6b0I$g{*AAujET%e6x(40Q*l#ylkV>9q4{X?d(YCn$iVQo1sRii_gOnj<a;U)pOM(cE*vbkqnJ5Xn0rC_j0x)G7a(xRYH3!AGsyxF5f1mpW5)oQP!V-40K<sTjQL!B1Y09EkqgA1fG)`%VX?qmnRwf|CcBG$~M+_uj1wMR00fB)i%e~h>szvmJ?tSGf}Q<@2n=h_dba-_4C6`5LPOd#+&fEXq7lpDWIPfcDWBSiqMSrBPdpcq=B-e@sSNO?j?^h*S9gfN)AV}CvP}vbx)?h+5gTIt*F8`q_)#^f<`tiywur4HTPb%%^?Q16-fB+mTY;Sd<MwRD@jZ=S`p)EEqsx6lxYzMYaIm$u$=&IC^;34(I7%~X?q!$I#Ykk@&1!@SV{7qaK4KpD1f;!RZMkK+{1||TuS=azfak<lCh2u2>=eEfbg(DvL|QaBBrl45m$u5WxC!b27&jtzu0182nvj$0&if)#5NzXaLf5_x7P*DKU7i@z4QS+3?CK^dtU2Hz>Wm!izBZIV-rq2uv&~@8E-9d-+&cl`>+m*;!x#6I9V6>4CP+ul>6QuDW)1^-yqjTYX6X?CZ0DBV6QDaphp5nu%0M0rMb}dXRK}d7hBp@b=k}|%eNxpyW2hl2e>kvAy&i4UxjP1hYVaj$9!EeNrH?jU^XMPNg=omh8W%|bqOY0-kU+S5U<+RWIo|{n!*dH49v_GO&cIg_uSEY^p>m5(nRg2#~#_>rW9OVVQ==7cIiuelk?@^F@=Z=FVojrCre_=CQup!!U(s>!UV7oV~{5|nn;fNW!G6^<IH7>=*vu3?u73Sj}L^k^4ds$b|H^^I<l&2M82y4ujZ2KFhdK{w#f8=D_VnS+&yNp3lyAV_FawCj&0q`iE}3tH}xlJb|JD~EN@Oshu@?h93BtySR#jkQ_!KyjiVkpi@6FYiCSLZax7CzVy~J1evbv3AW3Y=X4i7Rn2rDdLW+ucUd7Iu00Ec=up$5eM`@dzvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
