# twitter-influencers

## Setting up
~~~ bash
sudo apt-get install python python-pip
sudo pip install -r requirements.txt
~~~

## Getting data
- Download data from https://archive.org/download/archiveteam-twitter-stream-2016-07/archiveteam-twitter-stream-2016-07.tar
- Extract to ../twitter-data/

## Preprocessing data
~~~ bash
mkdir data
python ./data_arxiv_preprocess.py
python ./data_user_extract.py
python ./data_user_followers.py get
python ./data_user_followers.py combine
python ./data_tweet.py filter
~~~

# Train LDA
~~~ bash
mkdir models
python lda.py prepare_corpus -t ./data/data_tweets_selected.pkl -c ./models/lda_sep_corpus.mm -d ./models/lda_sep_dict.dict
python lda.py train -c ./models/lda_sep_corpus.mm -d ./models/lda_sep_dict.dict -m ./models/lda_sep_model.model -k 20
~~~
for parameters detail run
~~~ bash
python lda.py -h
~~~

# Train Doc2Vec & K-means
~~~ bash
mkdir models
python doc_vec_cluster.py train_doc2vec -t ./data/data_tweets_selected.pkl -d ./models/doc2vec_sep_model.model
python doc_vec_cluster.py train_kmeans -d ./models/doc2vec_sep_model.model -c ./models/doc2vec_sep_model.kmeans -k 20
~~~
for parameters detail run
~~~ bash
python doc_vec_cluster.py -h
~~~

# Visualise LDA & Doc2Vec+K-means
~~~ bash
jupyter notebook
~~~
Then run the two notebook `visualise_lda.ipynb` and `visualise_doc2vec.ipynb`
