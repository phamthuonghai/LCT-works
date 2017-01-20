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

CONTENT_WORD_POS = ['a', 'v', 'n'] # Adjective, verb and noun

f_text = open('./bilingual_data/en-vi.txt', 'r')

f_align = open('./bilingual_data/en-vi.align', 'r')

res = defaultdict(lambda : defaultdict(int))
set2 = set()

def parse_text(raw_text):
    temp = raw_text.split('|||')
    l1 = temp[0].strip().split()
    l2 = temp[1].strip().split()
    return l1, l2

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
                set2.add(w2)
        except Exception as e:
            print(e)
            pass

f_text.close()
f_align.close()

with open('./bilingual_data/en-vi.rows', 'w') as f:
    f.write('\n'.join(res.keys()))

with open('./bilingual_data/en-vi.cols', 'w') as f:
    f.write('\n'.join(set2))

with open('./bilingual_data/en-vi.sm', 'w') as f:
    for k, v in res.iteritems():
        for k2, v2 in v.iteritems():
            f.write(k + '\t' + k2 + '\t' + str(v2) + '\n')
