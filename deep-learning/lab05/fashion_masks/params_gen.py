import argparse
import json
import itertools

params_set = {
    'batch_size': [128, 256],
    'learning_rate': [0.01, 0.001],
    'learning_rate_final': [None, 0.0001],
    'model': ['CNN'],
    'cnn': [
        'C-1-3-1-same,O,F,R-100',
        'C-10-3-1-same,C-1-3-1-same,O,F,R-100',
        'CB-10-3-1-same,CB-1-3-1-same,O,F,RB-100',
        'C-32-3-1-same,C-1-3-1-same,O,F,R-100',
        'CB-32-3-1-same,CB-1-3-1-same,O,F,RB-100',
        'CB-10-3-1-same,CB-10-3-1-same,CB-1-3-1-same,O,F,RB-100',
    ],
    'train_loss_coef': [0.1, 0.3, 0.7]
}


def is_valid(item):
    return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']


full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params_cnn.json", type=str, help="Param file path.")
args = parser.parse_args()
with open(args.params, 'w') as f:
    json.dump(full_set, f)
