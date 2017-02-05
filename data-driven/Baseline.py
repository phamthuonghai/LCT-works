
# coding: utf-8

# # Baseline algorithm
# BoW and Linear Regression as in http://www.anthology.aclweb.org/W/W16/W16-0404.pdf

# In[1]:

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:

df_train = pd.read_pickle('./data/df_train.pkl')
df_test = pd.read_pickle('./data/df_test.pkl')
print(df_train.index.size)
print(df_test.index.size)


# ## BOW -> Linear regression

# In[9]:

vectorizer = CountVectorizer(min_df=3)
regr = LinearRegression()

bow_ln_regr = Pipeline([('bow', vectorizer), ('linear_regr', regr)])

# Valence
bow_ln_regr.fit(df_train.text, df_train.val)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Valence: %f' % r2_score(df_test.val, pred))
print('MSE score for Valence: %f' % mean_squared_error(df_test.val, pred))

# Arousal
bow_ln_regr.fit(df_train.text, df_train.aro)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Arousal: %f' % r2_score(df_test.aro, pred))
print('MSE score for Arousal: %f' % mean_squared_error(df_test.aro, pred))


# ## BOW -> Ridge regression

# In[14]:

vectorizer = CountVectorizer(min_df=3)
regr = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=5)

bow_ln_regr = Pipeline([('bow', vectorizer), ('linear_regr', regr)])

# Valence
bow_ln_regr.fit(df_train.text, df_train.val)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Valence: %f' % r2_score(df_test.val, pred))
print('MSE score for Valence: %f' % mean_squared_error(df_test.val, pred))

# Arousal
bow_ln_regr.fit(df_train.text, df_train.aro)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Arousal: %f' % r2_score(df_test.aro, pred))
print('MSE score for Arousal: %f' % mean_squared_error(df_test.aro, pred))


# ## BOW 3-grams -> Ridge regression

# In[15]:

vectorizer = CountVectorizer(min_df=3, ngram_range=(1,3))
regr = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=5)

bow_ln_regr = Pipeline([('bow', vectorizer), ('linear_regr', regr)])

# Valence
bow_ln_regr.fit(df_train.text, df_train.val)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Valence: %f' % r2_score(df_test.val, pred))
print('MSE score for Valence: %f' % mean_squared_error(df_test.val, pred))

# Arousal
bow_ln_regr.fit(df_train.text, df_train.aro)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Arousal: %f' % r2_score(df_test.aro, pred))
print('MSE score for Arousal: %f' % mean_squared_error(df_test.aro, pred))


# ## TFIDF -> Ridge regression

# In[16]:

vectorizer = TfidfVectorizer(min_df=3)
regr = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=5)

tfidf_ln_regr = Pipeline([('tfidf', vectorizer), ('linear_regr', regr)])

# Valence
bow_ln_regr.fit(df_train.text, df_train.val)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Valence: %f' % r2_score(df_test.val, pred))
print('MSE score for Valence: %f' % mean_squared_error(df_test.val, pred))

# Arousal
bow_ln_regr.fit(df_train.text, df_train.aro)
pred = bow_ln_regr.predict(df_test.text)
print('R2 score for Arousal: %f' % r2_score(df_test.aro, pred))
print('MSE score for Arousal: %f' % mean_squared_error(df_test.aro, pred))

