from multiprocessing import Pool
import json
import re
import bz2
import datetime
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

def date_range(start_time, end_time):
    while start_time <= end_time:
        yield start_time
        start_time += datetime.timedelta(days=1)

def time_range(start_time, end_time):
    while start_time <= end_time:
        yield start_time
        start_time += datetime.timedelta(minutes=1)

# Process data from raw files
def xml_bz2_processor(this_date):
    tweets_json = []

    err = 0
    ok = 0

    # iterate through every single minutes in this_day
    for cur_time in time_range(this_date, this_date + datetime.timedelta(minutes=24*60-1)):
        try:
            with bz2.open('../twitter-data/%s.json.bz2'
                            % cur_time.strftime('%Y/%m/%d/%H/%M'), 'rt') as f:
                data_lines = f.readlines()

                for data_line in data_lines:
                    tmp = json.loads(data_line)
                    # take only english tweets
                    if 'text' in tmp and ('lang' in tmp and tmp['lang'] == 'en'):
                        tmp['text'] = [w for w in tknzr.tokenize(tmp['text'].lower()) if re.match('^#?\w+$', w)]
                        tweets_json.append([str(tmp['id']), tmp['text'], tmp['timestamp_ms'], str(tmp['user']['id'])])
            ok += 1
        except Exception as inst:
            err += 1

    tweets = pd.DataFrame(tweets_json, columns=['id', 'text', 'timestamp', 'user'])

    tweets.to_csv('./data_%s.pkl' % this_date.strftime('%Y-%m-%d'))
    print("Read %s: %d successes, %d failures" % (this_date.strftime('%Y-%m-%d'), ok, err))

if __name__ == '__main__':

    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    punc = set('!$%^&*()_-+=\|{}[]:;"\'<>,.?/')

    # year, month, day
    data_time_start = datetime.datetime(2016, 7, 1)
    data_time_end = datetime.datetime(2016, 7, 31)
    with Pool(3) as p:
        p.map(xml_bz2_processor, date_range(data_time_start, data_time_end))
