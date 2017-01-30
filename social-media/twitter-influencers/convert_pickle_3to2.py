'''
    to be run in python3
'''
import pandas as pd
import pickle
import sys


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('convert_pickle_3to2.py input_file output_file')
        exit()

    fi = open(sys.argv[1], 'rb')
    tmp = pickle.load(fi)
    fo = open(sys.argv[2], 'wb')
    pickle.dump(tmp, fo, 2)
