# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from tqdm import tqdm
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("data_en_preprocess input_file output_no_pos output_with_pos")
        exit()

    tknzr = TreebankWordTokenizer()
    wnl = WordNetLemmatizer()
    tagger = StanfordPOSTagger('./stanford-postagger/models/english-left3words-distsim.tagger',
                                path_to_jar='./stanford-postagger/stanford-postagger.jar',
                                encoding='utf8')

    with open(sys.argv[1], 'r') as f:
        fout_token = open(sys.argv[2], 'w')
        fout_pos = open(sys.argv[3], 'w')

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
        

        part-of-speech tags used in the Penn Treebank Project:
        1.  CC  Coordinating conjunction
        2.  CD  Cardinal number
        3.  DT  Determiner
        4.  EX  Existential there
        5.  FW  Foreign word
        6.  IN  Preposition or subordinating conjunction
        7.  JJ  Adjective
        8.  JJR Adjective, comparative
        9.  JJS Adjective, superlative
        10. LS  List item marker
        11. MD  Modal
        12. NN  Noun, singular or mass
        13. NNS Noun, plural
        14. NNP Proper noun, singular
        15. NNPS    Proper noun, plural
        16. PDT Predeterminer
        17. POS Possessive ending
        18. PRP Personal pronoun
        19. PRP$    Possessive pronoun
        20. RB  Adverb
        21. RBR Adverb, comparative
        22. RBS Adverb, superlative
        23. RP  Particle
        24. SYM Symbol
        25. TO  to
        26. UH  Interjection
        27. VB  Verb, base form
        28. VBD Verb, past tense
        29. VBG Verb, gerund or present participle
        30. VBN Verb, past participle
        31. VBP Verb, non-3rd person singular present
        32. VBZ Verb, 3rd person singular present
        33. WDT Wh-determiner
        34. WP  Wh-pronoun
        35. WP$ Possessive wh-pronoun
        36. WRB Wh-adverb


        '''
        ts = [tknzr.tokenize(line_en.strip().strip('"')) for line_en in tqdm(f)]

        # p = nltk.pos_tag(t, tagset='universal')
        ps = tagger.tag_sents(ts)

        for p in tqdm(ps):
            r = []
            t = []
            for pos in p:
                t.append(pos[0].lower().strip('.,!'))
                if pos[1][0] == 'J':
                    r.append(wnl.lemmatize(pos[0].lower().strip('.,!'), pos='a') + u'/A')
                elif pos[1][0] == 'V':
                    r.append(wnl.lemmatize(pos[0].lower().strip('.,'), pos='v') + u'/V')
                elif pos[1][0] == 'N':
                    r.append(wnl.lemmatize(pos[0].lower().strip('.,!'), pos='n') + u'/N')
                else:
                    r.append(pos[0].lower()+ u'/' + pos[1])

            fout_token.write(' '.join(t) + '\n')
            fout_pos.write(' '.join(r) + '\n')

        fout_token.close()
        fout_pos.close()
