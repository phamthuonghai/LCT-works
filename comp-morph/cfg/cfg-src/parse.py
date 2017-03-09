import nltk
from nltk.grammar import Nonterminal
from nltk.parse import EarleyChartParser
from nltk.tokenize import WordPunctTokenizer

grammar = nltk.data.load('grammar.cfg')
grammar._start = Nonterminal('ROOT')
parser = EarleyChartParser(grammar)

tokenizer = WordPunctTokenizer()

sum_cnt = 0
sen_cnt = 0

fo = open('parses.out', 'w')

with open('sentences.txt') as f:
    for sentence in f:
        tk_sentence = tokenizer.tokenize(sentence)
        pars = parser.parse(tk_sentence)
        cnt = 0
        for par in pars:
            fo.write(str(par))
            fo.write('\n\n')
            cnt += 1

        fo.write(str(cnt))
        fo.write('\n-------------------------------------\n')
        sum_cnt += cnt
        sen_cnt += 1

    fo.write('\nThere are on average %.2f parses per sentence.\n' % (float(sum_cnt)/sen_cnt))

fo.close()
