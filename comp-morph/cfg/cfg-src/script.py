import nltk
from nltk.parse import EarleyChartParser
from nltk.tokenize import WordPunctTokenizer

grammar = nltk.data.load('grammar.cfg')
parser = EarleyChartParser(grammar)

tokenizer = WordPunctTokenizer()

sum_cnt = 0
sen_cnt = 0

with open('../sentences.txt') as f:
    for sentence in f:
        tk_sentence = tokenizer.tokenize(sentence)
        pars = parser.parse(tk_sentence)
        cnt = 0
        for par in pars:
            print(par)
            print('\n')
            cnt += 1

        print(cnt)
        print('-------------------------------------')
        sum_cnt += cnt
        sen_cnt += 1

    print('There are on average %.2f parses per sentence.' % (float(sum_cnt)/sen_cnt))
