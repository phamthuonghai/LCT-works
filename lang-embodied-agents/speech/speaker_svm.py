import os
import numpy as np
import re

from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav

from python_speech_features import mfcc
from python_speech_features import delta

DATA_DIR = "./Audio"
K_FOLD = 5
VALIDATE_SIZE = 0.3


def get_file_lists(data_dir='.'):
    speaker_files_list = {}
    get_id = re.compile('Speaker (\d+)')
    speaker_ids = []

    for root, sub_folders, files in os.walk(data_dir):
        for file in files:
            full_path = os.path.join(root, file)
            speaker_id = get_id.findall(root)[-1]   # Get the final matched
            if speaker_id not in speaker_files_list:
                speaker_ids.append(speaker_id)
                speaker_files_list[speaker_id] = []

            speaker_files_list[speaker_id].append(full_path)

    speaker_ids = sorted(speaker_ids)
    print('Speaker IDs: %s' % str(speaker_ids))
    return [speaker_files_list[_id] for _id in speaker_ids]


def fold_split(data_set, data_labels, test_size=0.3):
    train_files, test_files, train_labels, test_labels = train_test_split(data_set, data_labels, test_size=test_size)
    train_files = np.array(train_files).flatten()
    test_files = np.array(test_files).flatten()
    train_labels = np.array(train_labels).flatten()
    test_labels = np.array(test_labels).flatten()

    return train_files, test_files, train_labels, test_labels


def assign_class_labels(list_of_class_labels):
    class_iter = 1
    labels_list = []
    for _set in list_of_class_labels:
        n_files = len(_set)
        labels = np.zeros([n_files, 1], dtype=int)
        labels[:] = class_iter
        labels = np.array(labels).flatten()
        labels_list.extend(labels.tolist())
        class_iter += 1
    return labels_list


def extract_mfccs(file_list, label_list, use_deltas=True):
    mfccs = {}
    # total_files = len(file_list)
    max_len = 0
    progress_counter = 1
    try:
        for idx, file in enumerate(file_list):
            # print("Extracting MFCCs: %.2f %% complete" % ((progress_counter/total_files)*100))
            (rate, sig) = wav.read(file)
            mfcc_feat = mfcc(sig, rate)
            if use_deltas:
                d_mfcc_feat = delta(mfcc_feat, 2)
                all_feat = np.concatenate((mfcc_feat, d_mfcc_feat), axis=1)
            else:
                all_feat = mfcc_feat

            # normalize the features
            mean = np.mean(all_feat, axis=0)
            all_feat -= mean
            if max_len < all_feat.shape[0]:
                max_len = all_feat.shape[0]

            mfcc_file_labels = np.empty(len(all_feat))
            mfcc_file_labels.fill(label_list[idx])

            mfccs[file] = {"data": all_feat, "labels": mfcc_file_labels}
            progress_counter += 1

        for f in mfccs.keys():
            mfccs[f]['data'] = np.pad(mfccs[f]['data'],
                                      ((0, max_len-mfccs[f]['data'].shape[0]), (0, 0)), 'constant')

        return mfccs
    except Exception as e:
        print(str(e))
        return {}


def main():
    # preprocess data files, get a list of files per speaker
    speaker_files = get_file_lists(data_dir=DATA_DIR)
    # assign numeric labels for each class
    data_labels = assign_class_labels(speaker_files)
    # combine all into one set
    data_set = [file for speaker in speaker_files for file in speaker]

    for use_deltas in [False, True]:
        # extract MFCCs for data_set
        mfccs = extract_mfccs(file_list=data_set, label_list=data_labels, use_deltas=use_deltas)
        for fold in range(0, K_FOLD):
            # split into train/test sets
            train_files, test_files, train_labels, test_labels = fold_split(data_set, data_labels,
                                                                            test_size=VALIDATE_SIZE)

            train_labels_mat = to_categorical(train_labels)
            train_data = np.stack([mfccs[f]['data'] for f in train_files])

            model = Sequential()
            model.add(LSTM(100, input_shape=train_data.shape[1:]))
            model.add(Dense(train_labels_mat.shape[1], activation='sigmoid'))

            model.compile(optimizer=Adagrad(), loss='categorical_crossentropy')
            model.fit(train_data, train_labels_mat, batch_size=20)

            results_mat = model.predict(np.stack([mfccs[f]['data'] for f in test_files]))
            results = np.argmax(results_mat, axis=1)

            # for gmm_k in GMM_K:
            #
            #     # train UBM
            #     ubm_data = [mfccs[f]['data'] for f in train_files]
            #     ubm = mixture.GaussianMixture(n_components=gmm_k, covariance_type='diag', warm_start=True,
            #                                   tol=1 / (len(ubm_data) * len(ubm_data)))
            #     ubm.fit(ubm_data)
            #
            #     # train models
            #     gmms = []
            #     # train a GMM for each speaker
            #     speaker_classes = [1, 2, 3, 4, 5, 6]
            #     for speaker in speaker_classes:
            #         ubm_copy = copy.deepcopy(ubm)
            #         # print("Training model for speaker: %d" % speaker)
            #         speaker_data = []
            #         for file in train_files:
            #             data = mfccs[file]
            #             if data["labels"][0] == speaker:
            #                 speaker_data.extend(data["data"])
            #         ubm_copy.fit(speaker_data)
            #         gmms.append(ubm_copy)
            #
            #     # test models
            #     results = []
            #     for _, file in enumerate(test_files):
            #         # eval file under all models
            #         model_results = np.zeros(6)
            #         for idx, model in enumerate(gmms):
            #             test_mfccs = mfccs[file]["data"]
            #             score = model.score(test_mfccs)
            #             model_results[idx] = score
            #         max_index, max_value = max(enumerate(model_results), key=operator.itemgetter(1))
            #         results.append((max_index + 1))
            #
            # compare results with ground truth
            error = np.mean(results != test_labels)
            print("Fold: %d, K: %d, Deltas: %s, Error: %.2f %%" % (fold, 0, str(use_deltas), error * 100))
            # print confusion matrix
            print(confusion_matrix(test_labels, results))

    print("Done")


if __name__ == '__main__':
    main()
