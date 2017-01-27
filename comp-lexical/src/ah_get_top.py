from collections import defaultdict

# This ad-hoc code is developed to extract
# top 5 highest and lowest freq words [1:1550]
# from core.sm

r = defaultdict(int)

with open('./demo/core.sm', 'r') as f:
    for pair in f:
        tmp = pair.split()
        if len(tmp) != 3:
            continue
        r[tmp[0]] += int(tmp[2])

res = sorted(r, key=r.get, reverse=True)[:1550]

with open('./demo/list_top5_highest.txt', 'w') as f:
    f.write('\n'.join(res[:5]) + '\n')

with open('./demo/list_top5_lowest.txt', 'w') as f:
    f.write('\n'.join(res[1550-5:1550]) + '\n')
