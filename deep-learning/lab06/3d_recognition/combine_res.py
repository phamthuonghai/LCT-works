import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


prev_res = {}

for f in glob.glob('./results/*.txt'):
    prev_res[f] = np.loadtxt(f, dtype=int)

# Duplicate the top-2 best
prev_res['./results/6R-32-4.txt_2'] = prev_res['./results/6R-32-4.txt']
prev_res['./results/5R-32-4.txt_2'] = prev_res['./results/5R-32-4.txt']
prev_res['./results/5R-32-4.txt_3'] = prev_res['./results/5R-32-4.txt']

df = pd.DataFrame.from_dict(prev_res)

res = df.mode(axis=1)[0]

np.savetxt('./results/res.res', res.values, fmt='%d')

for col in df:
    print('Res diff to %s = %d' % (col, sum(df[col] != res)))

col = './results/5R-32-4.txt'
for the_id in range(20):
    test = np.load('modelnet20-test.npz')
    mis_match_ids = df[col] != res
    test_sample = test['voxels'][mis_match_ids].squeeze()
    print('Ensemble: %d' % res[mis_match_ids].iloc[the_id])
    for col in df:
        print('%s: %d' % (col, df[col][mis_match_ids].iloc[the_id]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(test_sample[the_id], edgecolor='w')
    plt.show()
