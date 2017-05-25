from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

from util import load_ppmi, load_feature_fit, load_concept_fit, \
        load_concept_corr


p = ArgumentParser()
p.add_argument("--ppmi-file", required=True)
p.add_argument("--feature-fit-dir", required=True)
p.add_argument("--corr-file", help="Path to concept correlation file")

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


def plot_concept_corr(concept_ppmis, concept_corr):
    xs, ys = [], []
    for concept, (_, ppmis) in concept_ppmis.items():
        try:
            concept_corr_i = concept_corr[concept]
        except KeyError:
            print("Skipping ", concept)
            continue

        xs.append(concept_corr_i)
        ys.append(np.median(ppmis))

    plt.scatter(xs, ys)
    plt.xlabel("concept corr")
    plt.ylabel("median pmi across features")
    plt.show()


def build_clfs(features, feature_ppmis):
    clfs, metrics = {}, {}
    for feature in tqdm(features, desc="Training classifiers"):
        neg_pmis, pos_pmis = feature_ppmis[feature]

        X = np.concatenate((neg_pmis, pos_pmis))[:, np.newaxis]
        y = np.concatenate((np.zeros_like(neg_pmis), np.ones_like(pos_pmis)),
                           axis=0)
        clfs[feature] = LogisticRegression()
        clfs[feature].fit(X, y)

        metrics[feature] = f1_score(y, clfs[feature].predict(X))

    return clfs, metrics


def main():
    features, feature_ppmis, concept_ppmis = load_ppmi(args.ppmi_file)
    feature_fits, cats = load_feature_fit(args.feature_fit_dir)
    concept_fits = load_concept_fit(args.feature_fit_dir)

    # Learn a classifier for each feature
    clfs, metrics = build_clfs(features, feature_ppmis)
    for feature in clfs:
        print("%s\t%.5f" % (feature, metrics[feature]))

    # Normalize for visualization purposes
    feature_ppmis = normalize_feature_ppmis(feature_ppmis)

    plot_feature_fit(feature_ppmis, feature_fits, cats)
    plot_concept_fit(concept_ppmis, concept_fits)

    if args.corr_file is not None:
        concept_corr = load_concept_corr(args.corr_file)
        plot_concept_corr(concept_ppmis, concept_corr)


if __name__ == '__main__':
    main()
