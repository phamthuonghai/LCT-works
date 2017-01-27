# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from tqdm import tqdm
from collections import defaultdict

'''
The tagset used by Vitk is that of the Vietnamese treebank. There are 18 different tags:

Np - Proper noun
Nc - Classifier
Nu - Unit noun
N - Common noun
V - Verb
A - Adjective
P - Pronoun
R - Adverb
L - Determiner
M - Numeral
E - Preposition
C - Subordinating conjunction
CC - Coordinating conjunction
I - Interjection
T - Auxiliary, modal words
Y - Abbreviation
Z - Bound morphemes
X - Unknown
'''

def parse_text(raw_text):
    temp = raw_text.split('|||')
    l1 = temp[0].strip().split()
    l2 = temp[1].strip().split()
    return l1, l2

CONTENT_WORD_POS = ['a', 'v', 'n'] # Adjective, verb and noun

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("data_convert.py.py input_text input_align output_rows output_cols output_sm")
        exit()

    f_text = open(sys.argv[1], 'r')

    f_align = open(sys.argv[2], 'r')

    res = defaultdict(lambda : defaultdict(int))
    set1 = defaultdict(int)
    set2 = defaultdict(int)

    for (c_text, c_align) in tqdm(zip(f_text, f_align)):
        text1, text2 = parse_text(c_text)
        for pair in c_align.strip().split():
            align_pair = pair.split('-')
            if len(align_pair) != 2:
                continue

            try:
                w1 = text1[int(align_pair[0])]
                w2 = text2[int(align_pair[1])]
                if ((len(w1) > 2 and w1[-2] == '/' and w1[-1] in CONTENT_WORD_POS) and 
                    (len(w2) > 2 and w2[-2] == '/' and w2[-1] in CONTENT_WORD_POS)):
                    # res[w1[:-2]][w2[:-2]] += 1
                    # set2.add(w2[:-2])
                    res[w1][w2] += 1
                    set1[w1] += 1
                    set2[w2] += 1
            except Exception as e:
                print(e)
                pass

    f_text.close()
    f_align.close()

    # Get 1550 target words
    target_words = sorted(set1, key=set1.get, reverse=True)[:1550]
    # Get 10000 context words as feature
    context_words = sorted(set2, key=set2.get, reverse=True)[:10000]

    with open(sys.argv[3], 'w') as f:
        f.write('\n'.join(target_words))

    with open(sys.argv[4], 'w') as f:
        f.write('\n'.join(context_words))

    target_words = set(target_words)
    context_words = set(context_words)
    with open(sys.argv[5], 'w') as f:
        for k, v in res.iteritems():
            if k in target_words:
                for k2, v2 in v.iteritems():
                    if k2 in context_words:
                        f.write(k + '\t' + k2 + '\t' + str(v2) + '\n')
