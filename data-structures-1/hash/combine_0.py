import sys
from collections import defaultdict

res_cnt = defaultdict(float)
res_time = defaultdict(float)
cnt = defaultdict(float)

for line in sys.stdin:
    _id, res_c, res_t = line.strip().split(',')
    res_cnt[_id] += float(res_c)
    res_time[_id] += float(res_t)
    cnt[_id] += 1

for k in sorted(res_cnt.keys()):
    print("%s,%f,%f\n" % (k, res_cnt[k]/cnt[k], res_time[k]/cnt[k]))
