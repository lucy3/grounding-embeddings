from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


p = ArgumentParser()
p.add_argument("--ppmi-file", required=True)
p.add_argument("--feature-fit-dir", required=True)

args = p.parse_args()


def load_ppmi():
    feature_ppmis = defaultdict(list)
    concept_ppmis = defaultdict(list)

    with open(args.ppmi_file, "r") as ppmi_f:
        for line in ppmi_f:
            fields = line.strip().split("\t")
            if len(fields) < 3: continue

            feature, concept, ppmi = fields[:3]

            # Normalize feature name to match feature_fit output
            feature = feature.replace(" ", "_")
            feature_ppmis[feature].append(float(ppmi))

            concept_ppmis[concept].append(float(ppmi))

    return feature_ppmis, concept_ppmis


def load_feature_fit():
    feature_fits = {}
    feature_categories = {}

    with Path(args.feature_fit_dir, "features.txt").open() as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 4: continue

            feature, category, score = fields[0], fields[1], fields[3]
            feature_fits[feature] = float(score)
            feature_categories[feature] = category

    return feature_fits, feature_categories


def load_concept_fit():
    concept_fits = {}

    with Path(args.feature_fit_dir, "concepts.txt").open() as f:
        for line in f:
            concept, score = line.strip().split()
            concept_fits[concept] = float(score)

    return concept_fits


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_feature_fit(feature_ppmis, feature_fits, cats):
    all_cats = sorted(cats.values())
    cat_ids = {c: i for i, c in enumerate(all_cats)}
    cmap = plt.cm.get_cmap("Set1", len(all_cats))

    xs, ys, cs, ls = [], [], [], []
    for feature, ppmis in feature_ppmis.items():
        try:
            feature_fit = feature_fits[feature]
        except KeyError:
            print("Skipping ", feature)
            continue

        xs.append(np.median(ppmis))
        ys.append(feature_fit)
        cs.append(cmap(cat_ids[cats[feature]]))
        ls.append(feature)

    print(len(xs))

    plt.scatter(xs, ys, c=cs)
    plt.xlabel("avg pmi across concepts")
    plt.ylabel("feature fit")
    plt.show()


def plot_concept_fit(concept_ppmis, concept_fits):
    xs, ys, ls = [], [], []
    for concept, ppmis in concept_ppmis.items():
        try:
            concept_fit = concept_fits[concept]
        except KeyError:
            print("Skipping ", concept)
            continue

        xs.append(np.median(ppmis))
        ys.append(concept_fit)
        ls.append(concept)

    plt.scatter(xs, ys)
    plt.xlabel("avg pmi across features")
    plt.ylabel("concept fit")
    plt.show()


def main():
    feature_ppmis, concept_ppmis = load_ppmi()
    feature_fits, cats = load_feature_fit()
    concept_fits = load_concept_fit()

    plot_feature_fit(feature_ppmis, feature_fits, cats)
    plot_concept_fit(concept_ppmis, concept_fits)


if __name__ == '__main__':
    main()
