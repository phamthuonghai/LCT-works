'''
to prepare corpus:
    python lda.py prepare_corpus --tweets ./data/data_tweets_selected.pkl 
                --corpus ./data/lda_sep_corpus.mm --dictionary ./data/lda_sep_dict.dict
to train:
    python lda.py train --corpus ./data/lda_sep_corpus.mm --dictionary ./data/lda_sep_dict.dict
                --model ./data/lda_sep_model.model
to visualise:
    python lda.py visualise --corpus ./data/lda_sep_corpus.mm --dictionary ./data/lda_sep_dict.dict
                --model ./data/lda_sep_model.model
for more parameters:
    python lda.py -h
'''

import json
import re
import bz2
import datetime
import argparse
from timeit import default_timer as timer
import pandas as pd
from tqdm import tqdm

from gensim import models
from gensim.corpora import Dictionary, MmCorpus

import pyLDAvis.gensim as gensimvis
import pyLDAvis

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def prepare_corpus(tweets_file, corpus_file, dictionary_file, author_topic):
    print('Loading tweets from ' + tweets_file)
    stop_words = set(stopwords.words('english'))
    stop_words.add(u'rt')

    tweets = pd.read_pickle(tweets_file)

    if author_topic:
        tweets = tweets.groupby('user').agg({'text': 'sum'})

    dictionary = Dictionary(tweets['text'])
    stopword_ids = map(dictionary.token2id.get, stop_words)
    dictionary.filter_tokens(stopword_ids)
    dictionary.compactify()
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=None)
    dictionary.compactify()

    corpus = [dictionary.doc2bow(doc) for doc in tweets['text']]
    
    # print(corpus)
    print("Writing corpus to " + corpus_file)
    MmCorpus.serialize(corpus_file, corpus)
    # print(dictionary)
    print("Writing dictionary to " + dictionary_file)
    
    dictionary.save(dictionary_file)

def train(corpus_file, dictionary_file, model_file, no_topic, no_pass, no_worker):
    print('Loading corpus from ' + corpus_file)
    corpus = MmCorpus(corpus_file)
    print('Loading dictionary from ' + dictionary_file)
    dictionary = Dictionary.load(dictionary_file)

    print('Training model %d topics in %d passes with %d workers' % (no_topic, no_pass, no_worker))
    lda = models.ldamulticore.LdaMulticore(corpus=corpus, id2word=dictionary,
                                            num_topics=no_topic, passes=no_pass, workers=no_worker)

    print('Writing model to ' + model_file)
    lda.save(model_file)

def visualise(model_file, corpus_file, dictionary_file):
    print('Loading corpus from ' + corpus_file)
    corpus = MmCorpus(corpus_file)
    print('Loading dictionary from ' + dictionary_file)
    dictionary = Dictionary.load(dictionary_file)
    print('Loading model from ' + model_file)
    model = models.ldamulticore.LdaMulticore.load(model_file)

    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    pyLDAvis.display(vis_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task to do: prepare_corpus, train, visualise")
    parser.add_argument("-t", "--tweets", help="Tweets file to parse")
    parser.add_argument("-c", "--corpus", help="Corpus file")
    parser.add_argument("-d", "--dictionary", help="Dictionary file")
    parser.add_argument("-m", "--model", help="Model file")
    parser.add_argument("-k", "--topics_num", type=int, default=20,
                        help="Number of topics")
    parser.add_argument("-p", "--passes", type=int, default=20,
                        help="Number of passes")
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Number of workers")
    parser.add_argument("-a", "--author-topic", help="Author topic model flag",
                        action="store_true")
    args = parser.parse_args()

    if args.task == 'prepare_corpus':
        prepare_corpus(args.tweets, args.corpus, args.dictionary, args.author_topic)
    elif args.task == 'train':
        start = timer()
        train(args.corpus, args.dictionary, args.model, args.topics_num, args.passes, args.workers)
        print('Time elasped ', time() - start)
    elif args.task == 'visualise':
        visualise(args.model, args.corpus, args.dictionary)
