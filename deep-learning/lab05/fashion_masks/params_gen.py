import argparse
import json
import itertools

params_set = {
    'batch_size': [
        # 128,
        256,
    ],
    'model': ['CNN'],
    'cnn': [
        # 'CB-32-3-1,M-3-1;F,R-100;CB-1-3-1,M-3-1',
        'CB-10-3-1,M-3-1,CB-10-3-1,M-3-1;M-3-1,F,RD-100-0.5;CB-10-3-1,M-3-1,CB-1-3-1,M-3-1',
    ],
    'masks_loss_coef': [
        1,
    ],
}


def is_valid(item):
    # return item['learning_rate_final'] is None or item['learning_rate_final'] < item['learning_rate']
    return True


full_set = [x for x in (dict(zip(params_set, x)) for x in itertools.product(*params_set.values())) if is_valid(x)]

parser = argparse.ArgumentParser()
parser.add_argument("--params", default="params_cnn.json", type=str, help="Param file path.")
args = parser.parse_args()
with open(args.params, 'w') as f:
    json.dump(full_set, f)
