from nltk import tree
from nltk.grammar import CFG, Nonterminal

prod = []
prod_cnf = []
s = ''

print('Bulding tree from parsed sentences')
with open('parsed_sentences.txt') as f:
    sentences = list(f) + ['']
    for line in sentences:
        line = line.strip()
        if len(line)>0:
            if line[0] != '#':
                s += line
        elif len(s) > 0:
            t = tree.Tree.fromstring(s)
            prod += t.productions()
            t.chomsky_normal_form()
            prod_cnf += t.productions()
            s = ''

prod = set(prod)
prod_cnf = set(prod_cnf)

print('Writing CFG to file with %d productions' % len(prod))
grammar = CFG(Nonterminal('ROOT'), prod)
with open('grammar.cfg', 'w') as f:
    f.write('\n'.join([str(p) for p in grammar.productions()]))


print('Writing CFG (CNF) to file with %d productions' % len(prod_cnf))
grammar_cnf = CFG(Nonterminal('ROOT'), prod_cnf)
with open('grammar_cnf.cfg', 'w') as f:
    f.write('\n'.join([str(p) for p in grammar_cnf.productions()]))
