'''
to train doc2vec:
    python doc_vec_cluster.py train_doc2vec -t ./data/data_tweets_selected.pkl -d ./model/doc2vec_model_tune.model
'''
# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import argparse
import pickle
import pandas as pd
from timeit import default_timer as timer
from gensim import models
from sklearn.cluster import KMeans


def train_doc2vec(tweets_file, doc2vec_file, author_topic, workers, iterations, feature_size):
    print('Loading tweets from ' + tweets_file)
    tweets = pd.read_pickle(tweets_file)

    if author_topic:
        tweets = tweets.groupby('user').agg({'text': 'sum'})

    print('Writing %d tweets to tmp file' % len(tweets.index))

    tmp_file = doc2vec_file + '.docs'

    with open(tmp_file, 'w') as f_tmp:
        for item in tweets.text:
            f_tmp.write(' '.join(item) + '\n')

    sentences = models.doc2vec.TaggedLineDocument(tmp_file)
    # model = models.doc2vec.Doc2Vec(sentences, size=feature_size, window=10, min_count=5,
        # iter=iterations, workers=workers)
    model = models.doc2vec.Doc2Vec(sentences, dm=0, dbow_words=1, size=100, window=10, hs=0,
        negative=5, sample=1e-4, iter=20, min_count=1, workers=workers)

    print('Writing doc2vec model to ' + doc2vec_file)
    model.save(doc2vec_file)


def train_kmeans(doc2vec_file, cluster_file, topics_num):
    print("Loading Doc2Vec model")
    md_doc2vec = models.Doc2Vec.load(doc2vec_file)
    md_kmean = KMeans(n_clusters=topics_num, n_jobs=-1)

    print("Training K-means with %d clusters" % topics_num)
    kmean = md_kmean.fit(md_doc2vec.docvecs)

    print("Writing K-means result to " + cluster_file)
    with open(cluster_file, 'wb') as pickle_file:
        pickle.dump(kmean, pickle_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task to do: train_doc2vec, train_kmeans")
    parser.add_argument("-t", "--tweets", help="Tweets file to parse")
    parser.add_argument("-c", "--cluster", help="Cluster model file")
    parser.add_argument("-d", "--doc2vec", help="Doc2vec model file")
    parser.add_argument("-m", "--model", help="Model file")
    parser.add_argument("-k", "--topics_num", type=int, default=20,
                        help="Number of topics")
    parser.add_argument("-i", "--iterations", type=int, default=20,
                        help="Number of iterations")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("-f", "--feature_size", type=int, default=100,
                        help="Number of workers")
    parser.add_argument("-a", "--author-topic", help="Author topic model flag",
                        action="store_true")
    args = parser.parse_args()

    if args.task == 'train_doc2vec':
        start = timer()
        train_doc2vec(args.tweets, args.doc2vec, args.author_topic,
                        args.workers, args.iterations, args.feature_size)
        print('Time elasped ', timer() - start)
    elif args.task == 'train_kmeans':
        start = timer()
        train_kmeans(args.doc2vec, args.cluster, args.topics_num)
        print('Time elasped ', timer() - start)
