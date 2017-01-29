import pandas as pd
import glob

df_user = pd.DataFrame(columns=['user', 'timestamp'])

for file_name in glob.glob('./data/data_2016-07-*.pkl'):
    print('Reading data from ' + file_name)
    df_td = pd.read_pickle(file_name)
    df_tmp = df_td.groupby('user', as_index=False)['timestamp'].count()
    df_user = pd.concat([df_user, df_tmp]).groupby('user', as_index=False)['timestamp'].sum()

print("Writing users' tweets count to ./data_user_tweets_count.pkl")
df_user.to_pickle('./data/data_user_tweets_count.pkl')
