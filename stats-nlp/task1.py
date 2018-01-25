from collections import defaultdict
import math
import random


def entropy(data):
    # Bigram and unigram counts
    c2 = defaultdict(lambda: defaultdict(float))
    c1 = defaultdict(float)

    l_data = len(data)

    for i in range(l_data - 1):
        c2[data[i]][data[i+1]] += 1
        c1[data[i]] += 1
    c1[data[l_data-1]] += 1

    H = 0
    for i in c2:
        p1 = c1[i] / l_data
        for j in c2[i]:
            p2 = c2[i][j] / (l_data - 1)  # Joint probability P(i, j)
            H -= math.log2(p2/p1) * p2

    return H, 2 ** H


def get_vocab_and_chars(data):
    vocab = set(data)
    chars = set([c for w in data for c in w])
    return vocab, chars


def mess_up_word(data, vocab, prob):
    vocab = list(vocab)
    return [w if random.random() > prob else random.choice(vocab) for w in data]


def mess_up_char(data, chars, prob):
    chars = list(chars)

    def replace_char(s):
        return ''.join([c if random.random() > prob else random.choice(chars) for c in s])
    return [replace_char(w) for w in data]


def task1(file_path, codec='utf-8'):
    text = [line.strip() for line in open(file_path, encoding=codec)]
    print('H=%f, PX=%f\n' % entropy(text))

    vocab, chars = get_vocab_and_chars(text)
    probs = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]

    print('=== Mess up words ===\n')
    for prob in probs:
        print('--- %f ---\n' % prob)
        res = []
        for epoch in range(10):
            new_text = mess_up_word(text, vocab, prob)
            h, _ = entropy(new_text)
            print('%f\n' % h)
            res.append(h)
        print('min: %f, avg: %f, max: %f\n' % (min(res), max(res), sum(res)/len(res)))

    print('=== Mess up characters ===\n')
    for prob in probs:
        print('--- %f ---\n' % prob)
        res = []
        for epoch in range(10):
            new_text = mess_up_char(text, chars, prob)
            h, _ = entropy(new_text)
            print('%f\n' % h)
            res.append(h)
        print('min: %f, avg: %f, max: %f\n' % (min(res), max(res), sum(res)/len(res)))

if __name__ == '__main__':
    print('\n========================== TEXTEN1 ==========================\n')
    task1('TEXTEN1.txt')

    print('\n========================== TEXTCZ1 ==========================\n')
    task1('TEXTCZ1.txt', codec='iso-8859-2')
