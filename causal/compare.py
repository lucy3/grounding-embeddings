from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


p = ArgumentParser()
p.add_argument("--ppmi-file", required=True)
p.add_argument("--feature-fit-file", required=True)

args = p.parse_args()


def load_ppmi():
    feature_ppmis = defaultdict(list)

    with open(args.ppmi_file, "r") as ppmi_f:
        for line in ppmi_f:
            fields = line.strip().split("\t")
            feature, ppmi = fields[0], fields[2]
            # Normalize feature name to match feature_fit output
            feature = feature.replace(" ", "_")
            feature_ppmis[feature].append(float(ppmi))

    return feature_ppmis


def load_feature_fit():
    feature_fits = {}

    with open(args.feature_fit_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            feature, score = fields[0], fields[3]
            feature_fits[feature] = float(score)

    return feature_fits


def main():
    ppmis = load_ppmi()
    ff = load_feature_fit()

    xs, ys, ls = [], [], []
    for feature, ppmis in ppmis.items():
        try:
            feature_fit = ff[feature]
        except KeyError:
            print("Skipping ", feature)
            continue

        xs.append(np.median(ppmis))
        ys.append(feature_fit)
        ls.append(feature)

    print(len(xs))

    plt.scatter(xs, ys)
    plt.xlabel("avg pmi")
    plt.ylabel("feature fit")
    plt.show()


if __name__ == '__main__':
    main()
