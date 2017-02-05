
# coding: utf-8

# In[1]:

from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# In[18]:

from bokeh.charts import Bar, output_file, output_notebook, show
from bokeh.layouts import row, gridplot
output_notebook()


# ## Loading features & labels

# In[3]:

df_train = pd.read_pickle('./data/df_train.pkl').sort_index()
df_train_norm = pd.read_pickle('./data/df_train_norm.pkl').sort_index()
df_train_emo = pd.read_pickle('./data/df_train_emo.pkl').sort_index()
df_train_doc2vec = pd.read_pickle('./data/df_train_doc2vec.pkl').sort_index()


# In[4]:

df_test = pd.read_pickle('./data/df_test.pkl').sort_index()
df_test_norm = pd.read_pickle('./data/df_test_norm.pkl').sort_index()
df_test_emo = pd.read_pickle('./data/df_test_emo.pkl').sort_index()
df_test_doc2vec = pd.read_pickle('./data/df_test_doc2vec.pkl').sort_index()


# In[5]:

val_linear = defaultdict(lambda : defaultdict(float))
val_svr = defaultdict(lambda : defaultdict(float))
val_ridge = defaultdict(lambda : defaultdict(float))

aro_linear = defaultdict(lambda : defaultdict(float))
aro_svr = defaultdict(lambda : defaultdict(float))
aro_ridge = defaultdict(lambda : defaultdict(float))

def test_regr(regr, train_data, train_label, test_data, test_label, show_params=False):
    to_train = pd.concat(train_data, axis=1, join='inner')
    to_test = pd.concat(test_data, axis=1, join='inner')
    print('Train data shape: ')
    print(to_train.shape)
    print('Test data shape: ')
    print(to_test.shape)
    regr.fit(to_train, train_label)
    pred = regr.predict(to_test)
    t = r2_score(test_label, pred), mean_squared_error(test_label, pred)
    print t
    if show_params:
        print(regr.best_params_)
    return t


# ## Linear Regression

# In[6]:

print("=========== Valence ===========")
val_linear['r2']['all'], val_linear['mse']['all'] =     test_regr(LinearRegression(), [df_train_doc2vec, df_train_norm, df_train_emo],
              df_train.val, [df_test_doc2vec, df_test_norm, df_test_emo], df_test.val)
    
print("=========== Arousal ===========")
aro_linear['r2']['all'], aro_linear['mse']['all'] =     test_regr(LinearRegression(), [df_train_doc2vec, df_train_norm, df_train_emo],
              df_train.aro, [df_test_doc2vec, df_test_norm, df_test_emo], df_test.aro)


# ## Ridge

# ### All

# In[7]:

print("=========== Valence ===========")
val_ridge['r2']['all'], val_ridge['mse']['all'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_doc2vec, df_train_norm, df_train_emo], df_train.val,
              [df_test_doc2vec, df_test_norm, df_test_emo], df_test.val, show_params=True)
    
print("=========== Arousal ===========")
aro_ridge['r2']['all'], aro_ridge['mse']['all'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_doc2vec, df_train_norm, df_train_emo], df_train.aro,
              [df_test_doc2vec, df_test_norm, df_test_emo], df_test.aro, show_params=True)


# ### Drop

# In[8]:

print("=========== Valence ===========")
val_ridge['r2']['drop_doc2vec'], val_ridge['mse']['drop_doc2vec'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_norm, df_train_emo], df_train.val,
              [df_test_norm, df_test_emo], df_test.val, show_params=True)
    
print("=========== Arousal ===========")
aro_ridge['r2']['drop_doc2vec'], aro_ridge['mse']['drop_doc2vec'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_norm, df_train_emo], df_train.aro,
              [df_test_norm, df_test_emo], df_test.aro, show_params=True)


# In[9]:

print("=========== Valence ===========")
val_ridge['r2']['drop_emo'], val_ridge['mse']['drop_emo'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_doc2vec, df_train_norm], df_train.val,
              [df_test_doc2vec, df_test_norm], df_test.val, show_params=True)
    
print("=========== Arousal ===========")
aro_ridge['r2']['drop_emo'], aro_ridge['mse']['drop_emo'] =     test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
              [df_train_doc2vec, df_train_norm], df_train.aro,
              [df_test_doc2vec, df_test_norm], df_test.aro, show_params=True)


# In[10]:

for col in df_train_norm.columns:
    print("=========== " + col + " ===========")
    print("==== Valence ====")
    val_ridge['r2']['drop_%s' % col], val_ridge['mse']['drop_%s' % col] =         test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
                  [df_train_doc2vec, df_train_norm.drop(col, axis=1), df_train_emo], df_train.val,
                  [df_test_doc2vec, df_test_norm.drop(col, axis=1), df_test_emo], df_test.val,
                  show_params=True)

    print("==== Arousal ====")
    aro_ridge['r2']['drop_%s' % col], aro_ridge['mse']['drop_%s' % col] =         test_regr(GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 1.0, 10.0]}, cv=5, n_jobs=-1,),
                  [df_train_doc2vec, df_train_norm.drop(col, axis=1), df_train_emo], df_train.aro,
                  [df_test_doc2vec, df_test_norm.drop(col, axis=1), df_test_emo], df_test.aro,
                  show_params=True)


# In[25]:

val_ridge['r2']['bow'] = 0.311061
val_ridge['mse']['bow'] = 0.015209
aro_ridge['r2']['bow'] = -0.033189
aro_ridge['mse']['bow'] = 0.058824


# In[26]:

result_ridge_val = pd.DataFrame.from_dict(val_ridge)
result_ridge_aro = pd.DataFrame.from_dict(aro_ridge)


# In[16]:

output_file('features.html')
p_val_r2 = Bar(result_ridge_val, values='r2', legend=None, color='blue',
               bar_width=0.7, title="(v1) Valence with R2 score")
p_val_mse = Bar(result_ridge_val, values='mse', legend=None, color='blue',
                bar_width=0.7, title="(v2) Valence with MSE")
p_aro_r2 = Bar(result_ridge_aro, values='r2', legend=None, color='orange',
               bar_width=0.7, title="(a1) Arousal with R2 score")
p_aro_mse = Bar(result_ridge_aro, values='mse', legend=None, color='orange',
                bar_width=0.7, title="(a2) Arousal with MSE")
show(gridplot([[p_val_r2, p_aro_r2], [p_val_mse, p_aro_mse]], plot_width=450, plot_height=400))


# In[27]:

result_ridge_val.join(result_ridge_aro, lsuffix='_val', rsuffix='_aro')


# ## SVR

# In[8]:

get_ipython().magic(u'timeit')
val_svr['r2']['all'], val_svr['mse']['all'] =     test_regr(GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, n_jobs=-1,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)}),
              [df_train_doc2vec, df_train_norm, df_train_emo],
              df_train.val, [df_test_doc2vec, df_test_norm, df_test_emo], df_test.val)

