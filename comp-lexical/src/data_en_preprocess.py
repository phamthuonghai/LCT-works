# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from tqdm import tqdm
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

tknzr = TreebankWordTokenizer()
wnl = WordNetLemmatizer()

with open('./bilingual_data/OpenSubtitles2016.en-vi.en', 'r') as f:
    fout_token = open('./bilingual_data/en.token.txt', 'w')
    fout_pos = open('./bilingual_data/en.pos.txt', 'w')

    '''
    Universal Part-of-Speech Tagset

    Tag     Meaning             English Examples
    ADJ     adjective           new, good, high, special, big, local
    ADP     adposition          on, of, at, with, by, into, under
    ADV     adverb              really, already, still, early, now
    CONJ    conjunction         and, or, but, if, while, although
    DET     determiner, article the, a, some, most, every, no, which
    NOUN    noun                year, home, costs, time, Africa
    NUM     numeral             twenty-four, fourth, 1991, 14:24
    PRT     particle            at, on, out, over per, that, up, with
    PRON    pronoun             he, their, her, its, my, I, us
    VERB    verb                is, say, told, given, playing, would
    .       punctuation marks   . , ; !
    X       other               ersatz, esprit, dunno, gr8, univeristy
    '''
    for line_en in tqdm(f):
        t = tknzr.tokenize(line_en.strip().strip('"'))
        fout_token.write(' '.join(t) + '\n')
        p = nltk.pos_tag(t, tagset='universal')
        r = []
        for pos in p:
            if pos[1] == 'ADJ':
                r.append(wnl.lemmatize(pos[0].lower().strip('.,!'), pos='a') + u'/A')
            elif pos[1] == 'VERB':
                r.append(wnl.lemmatize(pos[0].lower().strip('.,'), pos='v') + u'/V')
            elif pos[1] == 'NOUN':
                r.append(wnl.lemmatize(pos[0].lower().strip('.,!'), pos='n') + u'/N')
            else:
                r.append(pos[0].lower()+ u'/' + pos[1])

        fout_pos.write(' '.join(r) + '\n')

    fout_token.close()
    fout_pos.close()