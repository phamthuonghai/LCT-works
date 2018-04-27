import argparse
import json
import itertools

params_set = {
    'batch_size': [
        32,
    ],
    'model': [
        {
            'name': 'CLE',
            'cle_dim': 32,
            'rnn_cell': 'LSTM',
            'rnn_cell_dim': 64,
            'we_dim': 64,
        },
        {
            'name': 'CLE',
            'cle_dim': 32,
            'rnn_cell': 'GRU',
            'rnn_cell_dim': 64,
            'we_dim': 64,
        }
    ],
    'dropout': [
        0,
        0.5,
    ],
    'learning_rate': [
        0.01,
    ],
    'min_learning_rate': [
        1e-3,
    ],
    'lr_drop_max': [
        3,
    ],
    'lr_drop_rate': [
        0.7,
    ],
    'early_stop': [
        10,
    ]
}


def is_valid(item):
    return item['min_learning_rate'] is None or item['min_learning_rate'] < item['learning_rate']


full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params.json", type=str, help="Param file path.")
args = parser.parse_args()
with open(args.params, 'w') as f:
    json.dump(full_set, f)
