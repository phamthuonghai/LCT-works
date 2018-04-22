import argparse
import json
import itertools

params_set = {
    'pretrained': [
        'nasnet',
        # 'inception_v3',
    ],
    'batch_size': [
        # 16,
        32,
        # 64,
    ],
    'model': [
        '',
        'D-0.5',
        'D-0.5;RB-20;D-0.5',
        # 'RB-20;RB-20',
        # 'R-300;R-300',
        # 'R-300;R-300;R-300;R-300',
        # 'R-500;R-500;R-500;R-500',
    ],
    'clip_gradient': [
        0,
        # 100,
    ],
    'warmup': [
        # 10,
        # 20,
        40,
        60,
        # 1000,
    ]
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
