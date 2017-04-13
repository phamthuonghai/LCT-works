from collections import defaultdict

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tree import Tree


def parse(sen, grammar):
    l = len(sen)

    # DP mem f[i][j] = {label: [(sep_pos, prod)...]...}
    f = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Fill with anchor
    for i in range(0, l):
        cur_prod = grammar.productions(rhs=sen[i])
        for prod in cur_prod:
            if prod.lhs() not in f[i][i+1]:
                f[i][i+1][prod.lhs()] = [(None, prod)]

    # Main DP
    # Generate triplet (i, j, k)
    for _l in range(2, l+1):
        for i in range(0, l-_l+1):
            j = i+_l
            for k in range(i+1, j):

                # Find all production rules match with (i, k) and (k, j)
                for lhs_1 in f[i][k].keys():
                    for _prod in set(grammar.productions(rhs=lhs_1)):
                        if _prod.rhs()[1] in f[k][j]:

                            # Found
                            f[i][j][_prod.lhs()].append((k, _prod))

    return f


def back_track(prs_tbl, l, grammar):

    def go(i, j, term):
        res = []
        for k, _prod in prs_tbl[i][j][term]:
            if k is not None:
                # Find all possible trees on the left and right hand side
                l_trees = go(i, k, _prod.rhs()[0])
                r_trees = go(k, j, _prod.rhs()[1])

                # Combine left trees and right trees
                res += [Tree(str(_prod.lhs()), [_lt, _rt]) for _lt in l_trees for _rt in r_trees]
            else:
                res.append(Tree(str(_prod.lhs()), [str(_prod.rhs()[0])]))

        return res

    if l in prs_tbl[0] and grammar.start() in prs_tbl[0][l]:
        # Start back track with grammar start non-terminal
        res_trees = go(0, l, grammar.start())
        print('Number of trees: %d' % len(res_trees))
        for _tr in res_trees:
            _tr.pretty_print()
    else:
        print('Parsed failed')


def main():
    grammar = nltk.data.load('./grammar_cnf.cfg')

    if not (grammar.is_binarised() and grammar.is_chomsky_normal_form()):
        print('Grammar is not in CNF')
        return

    tokenizer = WordPunctTokenizer()
    with open('./sentences.txt') as f:
        for line in f:
            tk_sentence = tokenizer.tokenize(line)
            print('\n---- Parsing: %s' % line)
            parse_table = parse(tk_sentence, grammar)
            back_track(parse_table, len(tk_sentence), grammar)

if __name__ == '__main__':
    main()
