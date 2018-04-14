import argparse
import json
import itertools

params_set = {
    'modelnet_dim': [
        20,
        # 32,
    ],
    'train_split': [
        # 0.1,
        0.2,
    ],
    'batch_size': [
        16,
        32,
        # 64,
    ],
    'model': [
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300',
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300,R-300',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500,R-500',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-300,R-300,R-300,R-300,R-300,R-300',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-500,R-500,R-500,R-500,R-500,R-500',
    ],
    'bagging': [
        1,
    ],
}


def is_valid(item):
    # return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']
    return True


full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params.json", type=str, help="Param file path.")
args = parser.parse_args()
with open(args.params, 'w') as f:
    json.dump(full_set, f)
