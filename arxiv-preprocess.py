from tqdm import tqdm
import json
import re
import bz2
import datetime
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

# year, month, day, hour, minute
data_time_start = datetime.datetime(2016, 7, 1, 0, 0)
data_time_end = datetime.datetime(2016, 7, 31, 23, 59)

def time_range(start_time, end_time):
    while start_time <= end_time:
        start_time += datetime.timedelta(minutes=1)
        yield start_time

# Process data from raw files

tweets_json = []
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
punc = set('!$%^&*()_-+=\|{}[]:;"\'<>,.?/')

for cur_time in tqdm(time_range(data_time_start, data_time_end)):
    try:
        with bz2.open('../twitter-data/%s.json.bz2'
                        % cur_time.strftime('%Y/%m/%d/%H/%M'), 'rt') as f:
            data_lines = f.readlines()

            for data_line in data_lines:
                tmp = json.loads(data_line)
                if 'text' in tmp and ('lang' in tmp and tmp['lang'] == 'en'):
                    tmp['text'] = [w for w in tknzr.tokenize(tmp['text'].lower()) if re.match('^#?\w+$', w)]
                    tweets_json.append([tmp['id'], tmp['text'], tmp['timestamp_ms'], tmp['user']['id']])
    except Exception as inst:
        print(inst)

tweets = pd.DataFrame(tweets_json, columns=['id', 'text', 'timestamp', 'user'])

tweets.to_pickle('./data_%s_%s.pkl' % (data_time_start.strftime('%Y-%m-%d-%H-%M'),
                                              data_time_end.strftime('%Y-%m-%d-%H-%M')))
