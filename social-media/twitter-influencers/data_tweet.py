import pandas as pd
import argparse
import glob


def filter_tweets(input_file, output_file):
    df_users = pd.read_pickle(input_file)
    df_users.drop('tweets_count', axis=1, inplace=True)

    df_data = pd.DataFrame()
    for file_name in glob.glob('./data/data_2016-07-*.pkl'):
        print('Reading data from ' + file_name)
        df_td = pd.read_pickle(file_name)
        df_td = df_td.join(df_users, on='user', how='inner')
        df_data = pd.concat([df_data, df_td])

    df_data.reset_index(inplace=True)
    print('Writing selected tweets to ' + output_file)
    df_data.to_pickle(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="task to do: filter (filter tweets with selected users)")

    parser.add_argument("-i", "--input", help="selected users list (in pandas pickle)")
    parser.add_argument("-o", "--output", help="output file name")

    args = parser.parse_args()
    if args.task == 'filter':
        if not args.input:
            args.input = './data/data_user_selected.pkl'
        if not args.output:
            args.output = './data/data_tweets_selected.pkl'
        filter_tweets(args.input, args.output)
