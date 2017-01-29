try:
    import json
except ImportError:
    import simplejson as json

import argparse
from data_twitter_token import *
import pandas as pd
import tweepy
import sys
import pickle
import glob

def get_followers(start_after):
    if start_after:
        print("Start after " + start_after)
        file_suffix = start_after
    else:
        print("Start at begining")
        file_suffix = '0'


    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


    df_user = pd.read_pickle('./data/data_user_tweets_count.pkl')

    df_user.sort_values('timestamp', ascending=False, inplace=True)
    set_user = set(df_user.user)

    res = {}
    for user in df_user.user:
        try:
            if start_after:
                if user == start_after:
                    start_after = None
                continue

            print("Retrieving followers for user " + user)
            ids = []
            for page in tweepy.Cursor(api.followers_ids, user_id=user).pages():
                ids.extend(page)

            ids = set(ids)
            if len(ids) > 0:
                tmp = ids.intersection(set_user)
                res[user] = (tmp, ids.difference(tmp))
                
                if len(res.keys()) >= 100:
                    print("Writing followers to file after " + file_suffix + " to " + user)
                    with open('./data/part_user_followers_'
                            + file_suffix + '_' + user + '.pkl', 'wb') as pickle_file:
                        pickle.dump(res, pickle_file)
                    res = {}
                    file_suffix = user
        except Exception as e:
            print('Error! Passed user ' + user + ': ' + str(e))

def combine_result():
    dict_data = {}
    for file_name in glob.glob('./data/part_user_followers_*.pkl'):
        with open(file_name, 'rb') as pickle_file:
            dict_tmp = pickle.load(pickle_file)
            dict_data.update(dict_tmp)

    df_data = pd.DataFrame.from_dict(dict_data, orient='index')
    df_data.columns = ['inner_group', 'outer_group']

    df_user = pd.read_pickle('./data/data_user_tweets_count.pkl')
    df_user.set_index('user', inplace=True)
    df_user.columns = ['tweets_count']

    df_res = pd.merge(df_data, df_user, left_index=True, right_index=True, how='inner')
    df_res.sort_values('tweets_count', ascending=False, inplace=True)
    df_res.to_pickle('./data/data_user_selected_full.pkl')
    df_res[['tweets_count']].to_pickle('./data/data_user_selected.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task to do: get or combine")
    parser.add_argument("-s", "--start-after",
        help="retreive followers after user with specified id")
    args = parser.parse_args()

    if args.task == 'get':
        get_followers(args.start_after)
    elif args.task == 'combine':
        combine_result()
