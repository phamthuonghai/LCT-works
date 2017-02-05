
# coding: utf-8

# # Features extraction

# In[1]:

import pandas as pd
import pickle


# ## "Normal" features
# * Number of words with all capital characters
# * Number of capital characters
# * Number of characters
# * Number of words
# * Punctuations (Exclamation mark, Question mark)
# * Elongated words
# 

# In[2]:

dict_lex = {}
with open('./data/dict_lex.pkl', 'rb') as lex_file:
    dict_lex = pickle.load(lex_file)

print(len(dict_lex))


# In[3]:

def extract_lex(sent, dict_lex):
    res = 0
    for word in sent.strip().split():
        if word.lower() in dict_lex:
            res += dict_lex[word.lower()]
    return res

def extract_full_cap(sent):
    res = 0
    for word in sent.strip().split():
        if len(word) > 1 and word.isupper() and             word not in ['<PERSON>', '<URL>', '<ADDRESS>', '<PHONE>']:
            res += 1
    return res

def extract_char_cap(sent):
    res = 0
    for word in sent.strip().split():
        if word not in ['<PERSON>', '<URL>', '<ADDRESS>', '<PHONE>']:
            for char in word:
                if char.isupper():
                    res += 1
    return res

def extract_char_count(sent):
    return len(sent)

def extract_word_count(sent):
    return len(sent.strip().split())

def extract_punct(sent, punct):
    res = 0
    for char in sent:
        if char in punct:
            res += 1
    return res

def extract_elong(sent):
    res = 0
    for word in sent.strip().split():
        if len(word) > 2 and word[-1] == word[-2] and word[-2] == word[-3]:
            res += 1
    return res


# In[4]:

df_train = pd.read_pickle('./data/df_train.pkl')
df_train = df_train.assign(lex = df_train.apply(lambda row: extract_lex(row['text'], dict_lex), axis=1))
df_train = df_train.assign(full_cap = df_train.apply(lambda row: extract_full_cap(row['text']), axis=1))
df_train = df_train.assign(char_cap = df_train.apply(lambda row: extract_char_cap(row['text']), axis=1))
df_train = df_train.assign(char_count = df_train.apply(lambda row: extract_char_count(row['text']), axis=1))
df_train = df_train.assign(word_count = df_train.apply(lambda row: extract_word_count(row['text']), axis=1))
df_train = df_train.assign(punct_ex = df_train.apply(lambda row: extract_punct(row['text'], '!'), axis=1))
df_train = df_train.assign(punct_qt = df_train.apply(lambda row: extract_punct(row['text'], '%@#*&^?'), axis=1))
df_train = df_train.assign(elong_w = df_train.apply(lambda row: extract_elong(row['text']), axis=1))
print(df_train.head())
df_train.drop(['text', 'val', 'aro'], inplace=True, axis=1)
df_train.to_pickle('./data/df_train_norm.pkl')


# In[5]:

df_test = pd.read_pickle('./data/df_test.pkl')
df_test = df_test.assign(lex = df_test.apply(lambda row: extract_lex(row['text'], dict_lex), axis=1))
df_test = df_test.assign(full_cap = df_test.apply(lambda row: extract_full_cap(row['text']), axis=1))
df_test = df_test.assign(char_cap = df_test.apply(lambda row: extract_char_cap(row['text']), axis=1))
df_test = df_test.assign(char_count = df_test.apply(lambda row: extract_char_count(row['text']), axis=1))
df_test = df_test.assign(word_count = df_test.apply(lambda row: extract_word_count(row['text']), axis=1))
df_test = df_test.assign(punct_ex = df_test.apply(lambda row: extract_punct(row['text'], '!'), axis=1))
df_test = df_test.assign(punct_qt = df_test.apply(lambda row: extract_punct(row['text'], '%@#*&^?'), axis=1))
df_test = df_test.assign(elong_w = df_test.apply(lambda row: extract_elong(row['text']), axis=1))
df_test.drop(['text', 'val', 'aro'], inplace=True, axis=1)
df_test.to_pickle('./data/df_test_norm.pkl')


# ## Emoticons

# In[6]:

from sklearn.feature_extraction.text import CountVectorizer
import re

vectorizer = CountVectorizer(min_df=3, token_pattern='\S+')
re_emo = re.compile("([;:=][-\"']?[\*P)(\]><]+|[\*P)(\]<>]+[-\"']?[;:=]|lol|<3)", re.IGNORECASE)

def extract_emo(sent, re_emo):
    return ' '.join(re_emo.findall(sent))


# In[7]:

df_train = pd.read_pickle('./data/df_train.pkl')
df_train = df_train.assign(emo_tmp = df_train.apply(lambda row: extract_emo(row['text'].lower(), re_emo), axis=1))

train_emo = vectorizer.fit_transform(df_train.emo_tmp)

df_train_emo = pd.DataFrame(train_emo.todense())
df_train_emo.set_index(df_train.index, inplace=True)
df_train_emo.to_pickle('./data/df_train_emo.pkl')

df_test = pd.read_pickle('./data/df_test.pkl')
df_test = df_test.assign(emo_tmp = df_test.apply(lambda row: extract_emo(row['text'].lower(), re_emo), axis=1))

test_emo = vectorizer.transform(df_test.emo_tmp)

df_test_emo = pd.DataFrame(test_emo.todense())
df_test_emo.set_index(df_test.index, inplace=True)
df_test_emo.to_pickle('./data/df_test_emo.pkl')


# In[ ]:

print(df_train[df_train.index==1167])
print(df_train_emo[df_train_emo.index==1167])
print(df_test[df_test.index==1758])
print(df_test_emo[df_test_emo.index==1758])


# ## Doc2vec

# In[8]:

import gensim


# In[9]:

df_train = pd.read_pickle('./data/df_train.pkl')
df_test = pd.read_pickle('./data/df_test.pkl')


# In[10]:

sentences = []
for index, row in df_train.iterrows():
    sentences.append(gensim.models.doc2vec.TaggedDocument(row.text.split(), tags=[index]))


# In[11]:

# model = gensim.models.Doc2Vec(sentences, size=k, window=8, min_count=3, workers=4, iter=10)
model = gensim.models.Doc2Vec(sentences, dm=0, dbow_words=1, size=100, window=10, hs=0,
                              negative=5, sample=1e-4, iter=20, min_count=1, workers=4)

df_train_doc2vec = pd.DataFrame(list(model.docvecs))
df_train_doc2vec.drop([i for i in range(0, df_train_doc2vec.index.size) if i not in df_train.index],
                      inplace=True)

print("Train set size: %d" % df_train_doc2vec.index.size)

df_train_doc2vec.to_pickle('./data/df_train_doc2vec.pkl')

test_doc2vec = {}
for index, row in df_test.iterrows():
    test_doc2vec[index] = model.infer_vector(row.text.split())

df_test_doc2vec = pd.DataFrame.from_dict(test_doc2vec, orient='index')
print("Test set size: %d" %df_test_doc2vec.index.size)
df_test_doc2vec.to_pickle('./data/df_test_doc2vec.pkl')

