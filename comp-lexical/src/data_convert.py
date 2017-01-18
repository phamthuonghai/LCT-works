
from tqdm import tqdm
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import sys


lines = [None, None]

with open('./xces/en.txt', 'r') as f:
    lines[0] = [line.strip().split() for line in f.readlines()]

with open('./xces/vi.token.txt', 'r') as f:
    lines[1] = [line.strip().split() for line in f.readlines()]

align = []
with open('./xces/en-vi.align', 'r') as f:
    for line in f.readlines():
        tmp = []
        for pair in line.strip().split():
            tmp.append([int(id) for id in pair.split('-')])
        align.append(tmp)


lemmatizer = WordNetLemmatizer()


def word_matched(src_word, target_word, lang, pos):
#     return src_word == target_word
    if lang == 0:
        return src_word == lemmatizer.lemmatize(target_word, pos)
    else:
        return src_word == target_word

def words_in_line(word, line, lang, pos):
    return [idx for idx, w in enumerate(line) if word_matched(word, w, lang, pos)]

def find(word, lang, pos='n'):
    word = word.replace(' ', '_')
    res = []

    for idx, line in tqdm(enumerate(lines[lang])):
        word_ids = words_in_line(word, line, lang, pos)
        if word_ids and len(word_ids) > 1:
            for word_id in word_ids:
                tmp = [pair[1-lang] for pair in align[idx] if pair[lang] == word_id]
                if len(tmp) > 0:
                    res += [(idx, word_id, tmp)]
    
    return res


def print_lines(res_indexes, lang, max_lines = -1):
    display(Markdown('---'))
    cnt = 0

    for res_id in res_indexes:
        src_sen = ''
        dst_sen = ''
        
        for idx, w in enumerate(lines[lang][res_id[0]]):
            if idx == res_id[1]:
                src_sen += ' **' + w + '** '
            else:
                src_sen += ' ' + w + ' '

        for idx, w in enumerate(lines[1-lang][res_id[0]]):
            if idx in res_id[2]:
                dst_sen += ' **' + w + '** '
            else:
                dst_sen += ' ' + w + ' '
        
        display(Markdown(src_sen))
        display(Markdown(dst_sen))
        display(Markdown('---'))
        
        if max_lines > 0 and max_lines >= cnt:
            break

def print_freq(res_indexes, lang):
    res = defaultdict(int)
    for res_id in res_indexes:
        try:
            tmp_word = ' '.join([lines[1-lang][res_id[0]][idx] for idx in res_id[2]])
            res[tmp_word] += 1
        except:
            print("Unexpected error:", sys.exc_info()[0])
    print(sorted(list(res.items()), key=lambda x: x[1], reverse=True))
