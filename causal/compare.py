from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from util import load_ppmi, load_feature_fit, load_concept_fit


p = ArgumentParser()
p.add_argument("--ppmi-file", required=True)
p.add_argument("--feature-fit-dir", required=True)

args = p.parse_args()


def normalize_feature_ppmis(feature_ppmis):
    """
    Normalize PMI values collected for positive feature-concept associations
    by taking into account other negative feature-concept associations.
    """

    ret = {}
    for feature, (neg_scores, pos_scores) in feature_ppmis.items():
        ret[feature] = np.median(pos_scores) - np.median(neg_scores)

    return ret


def plot_feature_fit(feature_ppmis, feature_fits, cats):
    all_cats = sorted(cats.values())
    cat_ids = {c: i for i, c in enumerate(all_cats)}
    cmap = plt.cm.get_cmap("Set1", len(all_cats))

    xs, ys, cs, ls = [], [], [], []
    for feature, score in feature_ppmis.items():
        try:
            feature_fit = feature_fits[feature]
        except KeyError:
            print("Skipping ", feature)
            continue

        xs.append(score)
        ys.append(feature_fit)
        cs.append(cmap(cat_ids[cats[feature]]))
        ls.append(feature)

    print(len(xs))

    plt.scatter(xs, ys, c=cs)
    plt.xlabel("avg pmi across concepts (baselined by neg concepts)")
    plt.ylabel("feature fit")
    plt.show()


def plot_concept_fit(concept_ppmis, concept_fits):
    xs, ys, ls = [], [], []
    for concept, (_, ppmis) in concept_ppmis.items():
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
    feature_ppmis, concept_ppmis = load_ppmi(args.ppmi_file)
    feature_ppmis = normalize_feature_ppmis(feature_ppmis)

    feature_fits, cats = load_feature_fit(args.feature_fit_dir)
    concept_fits = load_concept_fit(args.feature_fit_dir)

    plot_feature_fit(feature_ppmis, feature_fits, cats)
    plot_concept_fit(concept_ppmis, concept_fits)


if __name__ == '__main__':
    main()
