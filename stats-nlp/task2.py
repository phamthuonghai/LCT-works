from collections import defaultdict
import math


EPSILON = 1e-7

# Trigram, bigram and unigram counts
c3 = c2 = c1 = None

# Conditional probabilities
p3 = p2 = p1 = None
p0 = 0

T = 0  # text size
V = 0  # vocab size


def get_counts_and_init_probs(data):
    # c3[w_i][w_i+1][w_i+2]
    global c3, p3, c2, p2, c1, p1, T, V, p0
    c3 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    p3 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # c2[w_i][w_i+1]
    c2 = defaultdict(lambda: defaultdict(int))
    p2 = defaultdict(lambda: defaultdict(float))
    # c1[w_i]
    c1 = defaultdict(int)
    p1 = defaultdict(float)

    T = len(data)       # text size
    V = len(set(data))  # vocab size
    p0 = 1. / V

    for i in range(T - 2):
        c3[data[i]][data[i+1]][data[i+2]] += 1

    for i in range(T - 1):
        c2[data[i]][data[i+1]] += 1

    for i in range(T):
        c1[data[i]] += 1


def get_prob(data):
    if data[2] not in p3[data[0]][data[1]]:
        vc3 = float(c3[data[0]][data[1]][data[2]])
        vc2f3 = sum(c3[data[0]][data[1]].values())  # c2 from c3
        p3[data[0]][data[1]][data[2]] = 1./V if vc2f3 == 0 else vc3/vc2f3

    if data[2] not in p2[data[1]]:
        vc2 = float(c2[data[1]][data[2]])
        vc1f2 = sum(c2[data[1]].values())  # c1 from c2
        p2[data[1]][data[2]] = 1./V if vc1f2 == 0 else vc2/vc1f2

    if data[2] not in p1:
        p1[data[2]] = float(c1[data[2]])/T

    return [p0, p1[data[2]], p2[data[1]][data[2]], p3[data[0]][data[1]][data[2]]]


def cross_entropy(l, data):
    # Data-oriented formula
    l_data = len(data)
    H = 0
    for i in range(l_data - 2):
        p = get_prob(data[i:i+3])
        p_prime = sum([l[i] * p[i] for i in range(4)])
        H -= math.log2(p_prime)

    return H / (l_data - 2)


def compute_lambdas(data):
    l_data = len(data)
    # Init lambdas
    # l = [.00001, .00009, .0009, .999]
    l = [.25, .25, .25, .25]

    while True:
        # Compute expected counts
        c_l = [0, 0, 0, 0]
        for i in range(l_data - 2):
            p = get_prob(data[i:i+3])
            p_prime = sum([l[i] * p[i] for i in range(4)])
            for j in range(4):
                c_l[j] += l[j] * p[j] / p_prime

        # Next lamdas
        sum_cl = sum(c_l)
        ln = [c_l[i] / sum_cl for i in range(4)]

        # Termination condition
        if max([abs(ln[i]-l[i]) for i in range(4)]) < EPSILON:
            break

        l = ln

    return ln


def l3_increase(l, incr):
    delta = (1-l[3]) * incr
    sum012 = l[0] + l[1] + l[2]
    return [l[0]-l[0]*delta/sum012, l[1]-l[1]*delta/sum012, l[2]-l[2]*delta/sum012, l[3] + delta]


def l3_shrink(l, dcr):
    delta = l[3] - l[3] * dcr
    sum012 = l[0] + l[1] + l[2]
    return [l[0]+l[0]*delta/sum012, l[1]+l[1]*delta/sum012, l[2]+l[2]*delta/sum012, l[3] * dcr]


def task2(file_path, codec='utf-8'):
    text = [line.strip() for line in open(file_path, encoding=codec)]
    test = text[-20000:]
    heldout = text[-60000:-20000]
    train = text[:-60000]

    get_counts_and_init_probs(train)

    print('--- Using training set ---\n')
    l = compute_lambdas(train)
    print('l0 = %f, l1 = %f, l2 = %f, l3 = %f\n' % (l[0], l[1], l[2], l[3]))
    print('H_test = %f\n' % cross_entropy(l, test))

    print('--- Using heldout set ---\n')
    l = compute_lambdas(heldout)
    print('l0 = %f, l1 = %f, l2 = %f, l3 = %f\n' % (l[0], l[1], l[2], l[3]))
    print('H_test = %f\n' % cross_entropy(l, test))

    print('--- Increasing ---\n')
    incrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for incr in incrs:
        print('%f\n' % incr)
        ln = l3_increase(l, incr)
        print('l0 = %f, l1 = %f, l2 = %f, l3 = %f\n' % (ln[0], ln[1], ln[2], ln[3]))
        print('H_test = %f\n' % cross_entropy(ln, test))

    print('--- Shrinking ---\n')
    dcrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dcrs.reverse()
    for dcr in dcrs:
        print('%f\n' % dcr)
        ln = l3_shrink(l, dcr)
        print('l0 = %f, l1 = %f, l2 = %f, l3 = %f\n' % (ln[0], ln[1], ln[2], ln[3]))
        print('H_test = %f\n' % cross_entropy(ln, test))

if __name__ == '__main__':
    print('\n========================== TEXTEN1 ==========================\n')
    task2('TEXTEN1.txt')

    # print('\n========================== TEXTCZ1 ==========================\n')
    task2('TEXTCZ1.txt', codec='iso-8859-2')
