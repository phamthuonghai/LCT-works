import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

word_align = ET.parse('./xces/en-vi.xml')
word_align_root = word_align.getroot()


def load_words(doc_path):
    res = defaultdict(list)
    with open(os.path.join('./xces', doc_path[:-3]), 'r') as f:
        from_doc = ET.fromstring(f.read())
        for sentence in from_doc.findall('s'):
            res[sentence.attrib['id']] = [word.text.lower() for word in sentence.findall('w')]
    return res


fout_en = open('./xces/en.txt', 'w')
fout_vi = open('./xces/vi.txt', 'w')

for film in tqdm(word_align_root):
    src_doc_sentences = load_words(film.attrib['fromDoc'])
    dst_doc_sentences = load_words(film.attrib['toDoc'])

    for pair in film:
        dw_id = pair.attrib['xtargets'].split(';')
        if len(dw_id) != 2 or dw_id[0] == '' or dw_id[1] == '':
            continue

        src_sen = ''
        dst_sen = ''

        for i in dw_id[0].split():
            src_sen += ' ' + ' '.join(src_doc_sentences[i])
        for i in dw_id[1].split():
            dst_sen += ' ' + ' '.join(dst_doc_sentences[i])
        
        src_sen = src_sen.strip()
        dst_sen = dst_sen.strip()
        
        if len(src_sen) < 1 or len(dst_sen) < 1:
            continue
        
        fout_en.write(src_sen + '\n')
        fout_vi.write(dst_sen + '\n')

fout_en.close()
fout_vi.close()


### Merge two raw text files to fast_align format

fin_en = open('./xces/en.txt', 'r')
fin_vi = open('./xces/vi.txt', 'r')
fout = open('./xces/en-vi-2.txt', 'w')

for (line_en, line_vi) in zip(fin_en, fin_vi):
    fout.write(line_en.strip() + ' ||| ' + line_vi.strip() + '\n')

fout.close()
fin_en.close()
fin_vi.close()
