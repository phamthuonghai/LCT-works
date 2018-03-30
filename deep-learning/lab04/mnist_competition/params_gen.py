import json
import itertools

params_set = {
    'batch_size': [100, 250, 500],
    'epochs': [100],
    'threads': [4],
    'learning_rate': [0.01, 0.001],
    'learning_rate_final': [None, 0.001, 0.0001],
    'cnn': [
        'CB-10-3-2-same,M-3-2,F,R-100',
        'C-10-3-2-same,M-3-2,F,R-100',
        'CB-32-3-3-valid,CB-32-3-3-valid,M-2-2,F,R-128',
        'CB-32-3-3-valid,CB-32-3-3-valid,M-2-2,F,RB-128',
    ],
}


def is_valid(item):
    return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']

full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

with open('params.json', 'w') as f:
    json.dump(full_set, f)
