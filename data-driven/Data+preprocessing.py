
# coding: utf-8

# # Data preprocess
# 
# ## Data acquisition
# * Main data from
# ~~~ bash
# wget http://mypersonality.org/wiki/lib/exe/fetch.php?media=dataset-fb-valence-arousal-anon.zip
# unzip fetch.php\?media\=dataset-fb-valence-arousal-anon.zip
# ~~~
# * Sentiment lexicon (http://saifmohammad.com/WebPages/SCL.html) with intensity
# ~~~ bash
# wget http://saifmohammad.com/WebDocs/lexiconstoreleaseonsclpage/SCL-NMA.zip
# unzip SCL-NMA.zip
# ~~~

# In[1]:

import pandas as pd
import nltk
import pickle
from nltk.tokenize.casual import TweetTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[2]:

df_raw = pd.read_csv('dataset-fb-valence-arousal-anon.csv')

df_raw.columns = ['text', 'val1', 'val2', 'aro1', 'aro2']
df_raw.dropna(inplace=True)
print(df_raw.index.size)
df_raw.head()


# ## Text
# To preserve emoticons, use TweetTokenizer instead of the recommended TreebankWordTokenizer

# In[3]:

tknzr = TweetTokenizer(reduce_len=True)
df_raw = df_raw.assign(text_tok = 
                       df_raw.apply(lambda row: ' '.join(tknzr.tokenize(row.text.strip())), axis=1))


# ## Label

# In[4]:

diff_val = df_raw['val1'] - df_raw['val2']
diff_aro = df_raw['aro1'] - df_raw['aro2']


# In[5]:

diff_aro.describe()


# In[6]:

df_raw = df_raw.assign(val = (df_raw['val1'] + df_raw['val2'] - 2.0)/(8*2))
df_raw = df_raw.assign(aro = (df_raw['aro1'] + df_raw['aro2'] - 2.0)/(8*2))


# In[7]:

df_raw.describe()


# In[8]:

df_raw.head()


# In[9]:

df_data = df_raw[['text_tok', 'val', 'aro']]
df_data.columns = ['text', 'val', 'aro']
df_raw.to_pickle('./data/df_raw.pkl')
df_data.to_pickle('./data/df_data.pkl')


# In[10]:

df_train, df_test = train_test_split(df_data, test_size=0.2)
print(df_train.index.size)
print(df_test.index.size)


# In[11]:

df_train.to_pickle('./data/df_train.pkl')
df_test.to_pickle('./data/df_test.pkl')


# ## Sentiment lexicon
# 
# Sentiment lexicon needed to be preprocessed to add contractions

# In[12]:

dict_lex = {}
with open('./SCL-NMA/SCL-NMA.txt', 'r') as lex_file:
    for line in lex_file:
        pair = line.strip().split()
        dict_lex[' '.join(pair[0:-1]).lower()] = float(pair[-1])

print("Original dict: %d entries" % len(dict_lex))

contractions_1 = {"would ": "'d ", "have ": "'ve ", "will ": "'ll ", "had ": "'d "}
contractions_2 = {"would have ": "would've ", "would have ": "'d've ", "will not ": "won't ",
                  "was not ": "wasn't ", "did not ": "didn't ", "could not ": "couldn't ",
                  "does not ": "doesn't ", "do not ": "don't ", "would not ": "wouldn't ",
                  "can not ": "cannot ", "cannot ": "can't ", "should not ": "shouldn't "}

new_dict = {}
for k, v in dict_lex.iteritems():
    for c_k, c_v in contractions_1.iteritems():
        if k.lower().startswith(c_k):
            new_dict[c_v + k[len(c_k):]] = v
    
    for c_k, c_v in contractions_2.iteritems():
        if k.lower().startswith(c_k):
            new_dict[c_v + k[len(c_k):]] = v

dict_lex.update(new_dict)

# Second time for "can not" "cannot" case
new_dict = {}
for k, v in dict_lex.iteritems():
    for c_k, c_v in contractions_1.iteritems():
        if k.lower().startswith(c_k):
            new_dict[c_v + k[len(c_k):]] = v
    
    for c_k, c_v in contractions_2.iteritems():
        if k.lower().startswith(c_k):
            new_dict[c_v + k[len(c_k):]] = v

dict_lex.update(new_dict)

# Third time for "' elimination"
new_dict = {}
for k, v in dict_lex.iteritems():
    if "'" in k:
        new_dict[''.join(k.split("'"))] = v

dict_lex.update(new_dict)

print("Modified dict: %d entries" % len(dict_lex))

with open('./data/dict_lex.pkl', 'wb') as lex_file:
    pickle.dump(dict_lex, lex_file)

