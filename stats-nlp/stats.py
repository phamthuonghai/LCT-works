from collections import defaultdict


# word count, number of characters (total, per word), the frequency of the most frequent words
# the number of words with frequency 1, etc.
def stats1(file_path, codec='utf-8'):
    text = [line.strip() for line in open(file_path, encoding=codec)]
    freqs = defaultdict(int)
    word_len = []

    for w in text:
        freqs[w] += 1
        word_len.append(len(w))

    most_freq_word = text[0]
    freq1_words = []

    for w in freqs:
        if freqs[w] > freqs[most_freq_word]:
            most_freq_word = w
        if freqs[w] == 1:
            freq1_words.append(w)

    print('Text size: %d\n' % len(text))
    print('Vocabulary size: %d\n' % len(freqs))
    print('Text size / vocabulary size: %f\n' % (1.0 * len(text)/len(freqs)))
    print('Characters count: %d\n' % sum(word_len))
    print('Word length: min %d avg %f max %d\n' % (min(word_len), sum(word_len) * 1.0 / len(word_len), max(word_len)))
    print('Most freq. word: %s %d\n' % (most_freq_word, freqs[most_freq_word]))
    print('%d (%f%%) word(s) with freq. = 1\n' % (len(freq1_words), 100.0 * len(freq1_words)/len(freqs)))


# coverage when splitting train, held-out and test sets
def stats2(file_path, codec='utf-8'):
    text = [line.strip() for line in open(file_path, encoding=codec)]
    test = text[-20000:]
    heldout = text[-60000:-20000]
    train = text[:-60000]

    train_unigram = set(train)
    train_bigram = set(['\t'.join(train[i:i+2]) for i in range(len(train)-1)])
    train_trigram = set(['\t'.join(train[i:i+3]) for i in range(len(train)-2)])

    heldout_unigram = set(heldout)
    heldout_bigram = set(['\t'.join(heldout[i:i+2]) for i in range(len(heldout)-1)])
    heldout_trigram = set(['\t'.join(heldout[i:i+3]) for i in range(len(heldout)-2)])
    
    test_unigram = set(test)
    test_bigram = set(['\t'.join(test[i:i+2]) for i in range(len(test)-1)])
    test_trigram = set(['\t'.join(test[i:i+3]) for i in range(len(test)-2)])

    print('Heldout coverage: unigram: %f, bigram %f, trigram %f\n' % (
        len(heldout_unigram.intersection(train_unigram))/len(heldout_unigram),
        len(heldout_bigram.intersection(train_bigram)) / len(heldout_bigram),
        len(heldout_trigram.intersection(train_trigram)) / len(heldout_trigram)
    ))

    print('Test coverage: unigram: %f, bigram %f, trigram %f\n' % (
        len(test_unigram.intersection(train_unigram))/len(test_unigram),
        len(test_bigram.intersection(train_bigram)) / len(test_bigram),
        len(test_trigram.intersection(train_trigram)) / len(test_trigram)
    ))


if __name__ == '__main__':
    print('\n========================== TEXTEN1 ==========================\n')
    stats1('TEXTEN1.txt')
    stats2('TEXTEN1.txt')

    print('\n========================== TEXTCZ1 ==========================\n')
    stats1('TEXTCZ1.txt', codec='iso-8859-2')
    stats2('TEXTCZ1.txt', codec='iso-8859-2')
