import glob


def compare(system, gold):
    if len(system) < len(gold):
        raise RuntimeError("The system output is shorter than gold data: {} vs {}.".format(len(system), len(gold)))

    correct = sum([gold[i] == system[i] for i in range(len(gold))])
    return correct * 100.0 / len(gold)


def combine(cans, d):
    to_count = [[] for _ in range(len(cans[0][d]))]
    for c in cans:
        for _id, row in enumerate(c[d]):
            to_count[_id].append(row)

    return [max(row, key=row.count) for row in to_count]


if __name__ == '__main__':
    with open('nli-dev.txt') as f:
        gold_dev = [t.strip().split('\t')[0] for t in f]

    candidates = []
    for dev_file in glob.glob('*/dev.txt'):
        with open(dev_file) as f:
            dev = [t.strip().split('\t')[0] for t in f]
        model = dev_file.split('/')[0]
        test_file = model + '/test.txt'
        with open(test_file) as f:
            test = [t.strip().split('\t')[0] for t in f]

        candidates.append({'model': model, 'dev': dev, 'test': test, 'dev_score': compare(dev, gold_dev)})

    candidates = sorted(candidates, key=lambda t: t['dev_score'], reverse=True)

    for can in candidates:
        print('%.5f -- %s' % (can['dev_score'], can['model']))

    print('\nEnsembling\n')

    max_dev_score = candidates[0]['dev_score']
    max_r_can = 1
    for r_can in range(len(candidates)):
        score = compare(combine(candidates[:r_can+1], 'dev'), gold_dev)
        print('%d -- %.5f' % (r_can, score))
        if score > max_dev_score:
            max_dev_score = score
            max_r_can = r_can

    print('Combining test output until #%d candidates' % max_r_can)

    with open('test.txt', 'w') as f:
        test = combine(candidates, 'test')
        f.write('\n'.join(test) + '\n')
