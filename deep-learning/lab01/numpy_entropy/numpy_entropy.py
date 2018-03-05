#!/usr/bin/env python3
from collections import defaultdict
import numpy as np


if __name__ == "__main__":
    # Load data distribution, each data point on a line
    word_count = defaultdict(int)
    total_count = 0
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            word_count[line] += 1
            total_count += 1

    # Create a NumPy array containing the data distribution
    data = []
    word_to_id = {}
    for _id, k in enumerate(sorted(word_count)):
        data.append(word_count[k] * 1.0 / total_count)
        word_to_id[k] = _id
    data = np.array(data)

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    model_dis = np.zeros(len(word_to_id))
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # process the line
            line = line.split()
            if line[0] in word_to_id:
                model_dis[word_to_id[line[0]]] = float(line[1])

    # Compute and print entropy H(data distribution)
    entropy = -np.sum(data * np.log(data))
    print("{:.2f}".format(entropy))

    # Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = -np.sum(data * np.log(model_dis))
    print("{:.2f}".format(cross_entropy))
    D_KL = cross_entropy - entropy
    print("{:.2f}".format(D_KL))
