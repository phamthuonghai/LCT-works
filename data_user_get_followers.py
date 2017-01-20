try:
    import json
except ImportError:
    import simplejson as json

from data_twitter_token import *
import pandas as pd
import tweepy
import sys
import pickle

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Start at begining")
        start_after = None
        file_suffix = '0'
    else:
        print("Start after " + sys.argv[1])
        start_after = sys.argv[1]
        file_suffix = start_after


    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


    df_user = pd.read_pickle('./data/data_user_tweets_count.pkl')

    df_user.sort_values('timestamp', ascending=False, inplace=True)
    set_user = set(df_user.user)

    res = {}
    for user in df_user.user:
        if start_after:
            if user == start_after:
                start_after = None
            continue

        print("Retrieving followers for user " + user)
        ids = []
        for page in tweepy.Cursor(api.followers_ids, user_id=user).pages():
            ids.extend(page)

        tmp_res = set_user.intersection(set(ids))
        if len(tmp_res) > 0:
            res[user] = tmp_res
            
            if len(res.keys()) > 100:
                print("Writing followers to file after " + file_suffix + " to " + user)
                pickle.dump(res, './data/part_user_followers_' 
                    + file_suffix + '_' + user + '.pkl')
                res = {}
                file_suffix = user
