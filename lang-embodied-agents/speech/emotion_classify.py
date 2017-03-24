#install Anaconda for Python, it has the packages you will need
#set Anaconda as your python interpreter

#Add python_speech_features to Anaconda:
# 1) cd C:\ProgramData\Anaconda3\Scripts
# 2) pip install python_speech_features

import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn import mixture
from sklearn.metrics import confusion_matrix
import scipy.io.wavfile as wav
import copy
import operator

from python_speech_features import mfcc
from python_speech_features import delta


def get_file_lists(main_dir="."):
    angry_list = []
    disappointed_list = []
    fear_list = []
    happy_list = []
    sad_list = []
    surprised_list = []

    for root, sub_folders, files in os.walk(main_dir):
        for file in files:
            full_path = os.path.join(root,file)
            if file.startswith("an"):
                angry_list.append(full_path)
            elif file.startswith("di"):
                disappointed_list.append(full_path)
            elif file.startswith("fe"):
                fear_list.append(full_path)
            elif file.startswith("ha"):
                happy_list.append(full_path)
            elif file.startswith("sa"):
                sad_list.append(full_path)
            elif file.startswith("su"):
                surprised_list.append(full_path)

    return angry_list,disappointed_list,fear_list,happy_list,sad_list,surprised_list


def fold_split(dataset, labels, test_size=0.3):
    train_files, test_files, train_labels, test_labels = train_test_split(data_set, data_labels, test_size=test_size)  # 30% for testing
    train_files = numpy.array(train_files).flatten()
    test_files = numpy.array(test_files).flatten()
    train_labels = numpy.array(train_labels).flatten()
    test_labels = numpy.array(test_labels).flatten()

    return train_files, test_files, train_labels, test_labels


def assign_class_labels(list_of_class_labels):
    class_iter = 1
    labels_list = []
    for set in list_of_class_labels:
        n_files = len(set)
        labels = numpy.zeros([n_files,1],dtype=int)
        labels[:] = class_iter
        labels = numpy.array(labels).flatten()
        labels_list.extend(labels.tolist())
        class_iter += 1
    return labels_list


def extract_mfccs(file_list,label_list,deltas=True):
    mfccs={}
    total_files = len(file_list)

    progress_counter = 1
    try:
        for idx, file in enumerate(file_list):
            #print("Extracting MFCCs: %.2f %% complete" % ((progress_counter/total_files)*100))
            (rate, sig) = wav.read(file)
            mfcc_feat = mfcc(sig, rate)
            if use_deltas:
                d_mfcc_feat = delta(mfcc_feat, 2)
                all_feat = numpy.concatenate((mfcc_feat,d_mfcc_feat), axis=1)
            else:
                all_feat = mfcc_feat

            # normalize the features
            mean = numpy.mean(all_feat,axis=0)
            all_feat = all_feat - mean

            mfcc_file_labels = numpy.empty(len(all_feat))
            mfcc_file_labels.fill(label_list[idx])

            mfccs[file] = {"data":all_feat, "labels":mfcc_file_labels}
            progress_counter += 1

        return mfccs
    except Exception as e:
        print(str(e))
        return {}


if __name__ == '__main__':
    # get files for each Emotion class in a separate list
    rootdir = "./RML_Emotion_Database/Audio"
    # use_deltas = True
    # k=64

    # preprocess data files, get a list of files per emotion
    angry_list, disappointed_list, fear_list, happy_list, sad_list, surprised_list = get_file_lists(main_dir=rootdir)

    # assign numeric labels for each class
    data_labels = assign_class_labels([angry_list, disappointed_list, fear_list, happy_list, sad_list, surprised_list])

    # combine all into one set
    data_set = angry_list + disappointed_list + fear_list + happy_list + sad_list + surprised_list
    del angry_list, disappointed_list, fear_list, happy_list, sad_list, surprised_list

    # split into train/test sets
    train_files, test_files, train_labels, test_labels = fold_split(data_set, data_labels, test_size=0.3)  # 30% for testing

    for use_deltas in [False]:
        # extract MFCCs for data_set
        mfccs = extract_mfccs(file_list=data_set, label_list=data_labels, deltas=use_deltas)
        for fold in range(0, 4):
            # for gmm_k in [16,32,64,128,256]:
            for gmm_k in [16]:

                # train UBM
                ubm_data=[]
                for file in train_files:
                    data=mfccs[file]
                    ubm_data.extend(data["data"])
                ubm = mixture.GaussianMixture(n_components=gmm_k, covariance_type='diag', warm_start=True, tol=1/(len(ubm_data)*len(ubm_data)))
                ubm.fit(ubm_data)

                # train models
                gmms=[]
                # train a GMM for each emotion
                emotion_classes = [1,2,3,4,5,6]
                for emotion in emotion_classes:
                    ubm_copy = copy.deepcopy(ubm)
                    #print("Training model for emotion: %d" % emotion)
                    emotion_data = []
                    for file in train_files:
                        data = mfccs[file]
                        if data["labels"][0] == emotion:
                            emotion_data.extend(data["data"])
                    ubm_copy.fit(emotion_data)
                    gmms.append(ubm_copy)

                # test models
                results = []
                for idx, file in enumerate(test_files):
                    #eval file under all models
                    model_results = numpy.zeros(6)
                    for idx, model in enumerate(gmms):
                        test_mfccs = mfccs[file]["data"]
                        score = model.score(test_mfccs)
                        model_results[idx] = score
                    max_index, max_value = max(enumerate(model_results), key=operator.itemgetter(1))
                    results.append((max_index+1))

                # compare results with ground truth
                error = numpy.mean(results != test_labels)
                print("Fold: %d, K: %d, Deltas: %s, Error: %.2f %%" % (fold, gmm_k, str(use_deltas), (error*100)))
                # print confusion matrix
                print(confusion_matrix(test_labels,results))

            # del mfccs

    print("Done")

# Fold: 0, K: 16, Deltas: False, Error: 51.39 %
# [[24  0  3  2  0  7]
#  [ 4 15  7  5  1  2]
#  [ 1  7 11  9 15  1]
#  [ 3  6  0 13  5  1]
#  [ 0  4  2  3 34  1]
#  [ 7  3  2  9  1  8]]
# Fold: 0, K: 32, Deltas: False, Error: 54.17 %
# [[22  0  6  1  0  7]
#  [ 3 15  7  7  2  0]
#  [ 1  7  9  9 17  1]
#  [ 3  6  3 11  3  2]
#  [ 0  4  6  3 31  0]
#  [ 3  3  4  9  0 11]]
# Fold: 0, K: 64, Deltas: False, Error: 48.61 %
# [[23  0  1  3  0  9]
#  [ 4 18  5  6  1  0]
#  [ 1  8  8  7 19  1]
#  [ 3  6  0 13  4  2]
#  [ 0  4  2  3 34  1]
#  [ 3  3  3  6  0 15]]
# Fold: 0, K: 128, Deltas: False, Error: 44.91 %
# [[22  0  2  4  0  8]
#  [ 3 18  5  6  1  1]
#  [ 1  8 10  6 17  2]
#  [ 1  6  0 15  4  2]
#  [ 0  3  3  3 34  1]
#  [ 3  0  3  4  0 20]]
# Fold: 0, K: 256, Deltas: False, Error: 46.76 %
# [[21  0  1  5  0  9]
#  [ 3 16  5  7  1  2]
#  [ 2  5 10  8 16  3]
#  [ 2  5  0 16  5  0]
#  [ 0  3  2  3 35  1]
#  [ 2  1  2  8  0 17]]
# Done


# My run
# /usr/local/lib/python3.5/dist-packages/sklearn/mixture/base.py:237: ConvergenceWarning: Initialization 1 did not converged. Try different init parameters, or increase max_iter, tol or check for degenerate data.
#   % (init + 1), ConvergenceWarning)
# Fold: 0, K: 16, Deltas: False, Error: 45.37 %
# [[20  0  1  3  0  3]
#  [ 2 19  5  6  3  1]
#  [ 2  6 16  3 11  3]
#  [ 5  3  6 21  4  1]
#  [ 0  4  4  3 29  0]
#  [ 4  1  6  6  2 13]]
# Fold: 0, K: 32, Deltas: False, Error: 48.61 %
# [[20  0  1  1  0  5]
#  [ 3 20  3  5  3  2]
#  [ 1  6 14  4 12  4]
#  [ 6  4 10 14  4  2]
#  [ 0  3  4  3 30  0]
#  [ 2  1  6  9  1 13]]
# Fold: 0, K: 64, Deltas: False, Error: 45.83 %
# [[21  0  0  2  0  4]
#  [ 2 21  4  5  3  1]
#  [ 1  5 17  3 12  3]
#  [ 6  2  5 17  7  3]
#  [ 0  4  6  3 27  0]
#  [ 2  1  7  7  1 14]]
# Fold: 0, K: 128, Deltas: False, Error: 44.44 %
# [[21  0  0  1  0  5]
#  [ 2 21  4  5  3  1]
#  [ 2  6 17  3 11  2]
#  [ 5  2  8 17  6  2]
#  [ 0  4  6  3 27  0]
#  [ 2  1  8  4  0 17]]
# Fold: 0, K: 256, Deltas: False, Error: 44.91 %
# [[21  0  0  1  0  5]
#  [ 2 21  3  6  4  0]
#  [ 2  6 15  3 13  2]
#  [ 5  1  8 18  4  4]
#  [ 0  4  6  3 27  0]
#  [ 2  1  4  6  2 17]]
# Done
