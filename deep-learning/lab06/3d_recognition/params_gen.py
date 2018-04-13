import argparse
import json
import itertools

params_set = {
    'modelnet_dim': [
        20,
        32,
    ],
    'train_split': [
        0.1,
        # 0.2,
    ],
    'batch_size': [
        128,
        # 512,
    ],
    'model': [
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-1000,R-1000,R-100',
        # 'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-1000,D-0.5,R-1000,D-0.5,R-100',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-100,R-100,R-100,R-100',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-100,D-0.5,R-100,D-0.5,R-100,D-0.5,R-100',
        'CB-32-5-1,M-5-1,CB-32-5-1,M-5-1,CB-32-3-1,M-3-1,F,R-100,R-100,R-100'
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
