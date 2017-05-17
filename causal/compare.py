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
    feature_categories = {}

    with open(args.feature_fit_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            feature, category, score = fields[0], fields[1], fields[3]
            feature_fits[feature] = float(score)
            feature_categories[feature] = category

    return feature_fits, feature_categories


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def main():
    ppmis = load_ppmi()
    ff, cats = load_feature_fit()

    all_cats = sorted(cats.values())
    cat_ids = {c: i for i, c in enumerate(all_cats)}
    cmap = plt.cm.get_cmap("Set1", len(all_cats))

    xs, ys, cs, ls = [], [], [], []
    for feature, ppmis in ppmis.items():
        try:
            feature_fit = ff[feature]
        except KeyError:
            print("Skipping ", feature)
            continue

        xs.append(np.median(ppmis))
        ys.append(feature_fit)
        cs.append(cmap(cat_ids[cats[feature]]))
        ls.append(feature)

    print(len(xs))

    plt.scatter(xs, ys, c=cs)
    plt.xlabel("avg pmi")
    plt.ylabel("feature fit")
    plt.show()


if __name__ == '__main__':
    main()
