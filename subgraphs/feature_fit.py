import codecs
from collections import defaultdict, namedtuple, Counter
from concurrent import futures
from pathlib import Path
from pprint import pprint
import csv
import os.path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from sklearn import metrics
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm, trange
import seaborn as sns

import domain_feat_freq
import get_domains
import random

# The "pivot" source is where we draw concept representations from. The
# resulting feature_fit metric represents how well these representations encode
# the relevant features. Each axis of the resulting graphs also involves the
# pivot source.
PIVOT = "wikigiga"
if PIVOT == "mcrae":
    INPUT = "./all/mcrae_vectors.txt"
elif PIVOT == "cslb":
    INPUT = "./all/cslb_vectors.txt"
elif PIVOT == "wikigiga":
    INPUT = "../glove/glove.6B.300d.txt"
elif PIVOT == "cc":
    INPUT = "../glove/glove.840B.300d.txt"

SOURCE = "cslb"
if SOURCE == "mcrae":
    FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
else:
    FEATURES = "../cslb/norms.dat"
VOCAB = "./all/vocab_%s.txt" % SOURCE
EMBEDDINGS = "./all/embeddings.%s.%s.npy" % (SOURCE, PIVOT)

OUTPUT = "./all/feature_fit/%s/%s.txt" % (SOURCE, PIVOT)
PEARSON1_NAME = "%s_%s" % (SOURCE,
                           PIVOT if PIVOT != SOURCE else "%s_wikigiga" % SOURCE)
PEARSON1 = './all/pearson_corr/%s/corr_%s.txt' % (SOURCE, PEARSON1_NAME)
PEARSON2_NAME = "wordnetres_%s" % PIVOT
PEARSON2 = './all/pearson_corr/%s/corr_%s.txt' % (SOURCE, PEARSON2_NAME)
GRAPH_DIR = './all/feature_fit/%s/%s' % (SOURCE, PIVOT)

Feature = namedtuple("Feature", ["name", "concepts", "wb_label", "wb_maj",
                                 "wb_min", "br_label", "disting"])


def load_embeddings(concepts):
    assert Path(VOCAB).is_file()
    with open(VOCAB, "r") as vocab_f:
        vocab = [line.strip() for line in vocab_f]

    if Path(EMBEDDINGS).is_file():
        embeddings = np.load(EMBEDDINGS)
        assert len(embeddings) == len(vocab), "%i %i" % (len(embeddings), len(vocab))
    else:
        embeddings = {}
        with open(INPUT, "r") as glove_f:
            for line in glove_f:
                fields = line.strip().split()
                word = fields[0]
                if word in concepts and word in vocab:
                    vec = [float(x) for x in fields[1:]]
                    embeddings[word] = vec

        embeddings = [embeddings[x] for x in vocab]

        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS, embeddings)

    return vocab, embeddings


def load_features_concepts():
    """
    Returns:
        features: string -> Feature
        concepts: set of strings
    """
    features = {}
    concepts = set()

    if SOURCE == "mcrae":
        with open(FEATURES, "r") as features_f:
            for line in features_f:
                fields = line.strip().split("\t")
                concept_name, feature_name = fields[:2]
                if concept_name == "Concept" or concept_name == "dunebuggy":
                    # Header row / row we are going to ignore!
                    continue
                if feature_name not in features:
                    features[feature_name] = Feature(
                            feature_name, set(), *fields[2:6], fields[10])

                features[feature_name].concepts.add(concept_name)
                concepts.add(concept_name)

        lengths = [len(f.concepts) for f in features.values()]
        print("# of features with particular number of associated concepts:")
        pprint(Counter(lengths))

    if SOURCE == "cslb":
        with open(FEATURES, "r") as features_f:
            reader = csv.DictReader(features_f, delimiter='\t')
            for row in reader:
                concept_name = row["concept"]
                feature_name = "_".join(row["feature"].split())
                if feature_name not in features:
                    features[feature_name] = Feature(feature_name, set(),
                        "", "", "", row["feature type"], "")

                features[feature_name].concepts.add(concept_name)
                concepts.add(concept_name)
        lengths = [len(f.concepts) for f in features.values()]
        print("# of features with particular number of associated concepts:")
        pprint(Counter(lengths))

    return features, concepts


def loocv_feature(C, X, y, f_idx, clf, n_concept_samples=5):
    """
    Evaluate LOOCV regression on a sampled feature subset for a given
    classifier instance.
    """
    scores = []

    # Find all concepts which (1) do or (2) do not have this feature
    c_idxs = y.nonzero()[0]
    c_not_idxs = (1 - y).nonzero()[0]

    n_f_concepts = min(len(c_idxs), n_concept_samples)
    f_concepts = np.random.choice(c_idxs, replace=False, size=n_f_concepts)
    for c_idx in f_concepts:
        X_loo = np.concatenate([X[:c_idx], X[c_idx+1:]])
        y_loo = np.concatenate([y[:c_idx], y[c_idx+1:]])

        clf_loo = clone(clf)
        clf_loo.fit(X_loo, y_loo)

        # Draw negative samples for a ranking loss
        test = np.concatenate([X[c_idx:c_idx+1], X[c_not_idxs]])
        pred_prob = clf_loo.predict_proba(X)[:, 1]

        score = np.log(pred_prob[0]) + np.mean(np.log(1 - pred_prob[1:]))
        scores.append(score)

    return C, scores


def analyze_features(features, word2idx, embeddings):
    """
    Compute metrics for all features.

    Arguments:
        features: dict of feature_name -> `Feature`
        word2idx: concept name -> concept id dict
        embeddings: numpy array of concept embeddings

    Returns:
        List of tuples with structure:
            feature_name:
            n_concepts: number of concepts which have this feature
            metric: goodness metric, where higher is better
    """

    # Prepare for a multi-label logistic regression.
    X = embeddings
    Y = np.zeros((len(word2idx), len(features)))

    feature_names = sorted(features.keys())
    for f_idx, f_name in enumerate(feature_names):
        feature = features[f_name]
        concepts = [word2idx[c] for c in feature.concepts if c in word2idx]
        if len(concepts) < 5:
            continue

        for c_idx in concepts:
            Y[c_idx, f_idx] = 1

    # # Sample a few random features.
    # # For the sampled features, we'll do LOOCV to evaluate each possible C.
    # nonzero_features = Y.sum(axis=0).nonzero()[0]

    # C_results = defaultdict(list)
    # with futures.ProcessPoolExecutor(10) as executor:
    #     C_futures = []

    #     C_choices = [10 ** exp for exp in range(-3, 1)]
    #     C_choices += [5 * (10 ** exp) for exp in range(-3, 1)]
    #     for C in C_choices:
    #         reg = LogisticRegression(class_weight="balanced", fit_intercept=False,
    #                                  C=C)

    #         for f_idx in nonzero_features:
    #             C_futures.append(executor.submit(loocv_feature,
    #                                              C, X, Y[:, f_idx], f_idx, reg))

    #     for future in tqdm(futures.as_completed(C_futures), total=len(C_futures),
    #                        file=sys.stdout):
    #         C, scores = future.result()
    #         C_results[C].extend(scores)

    # # Prefer stronger regularization; sort descending by metric, then by
    # # negative C
    # C_results = sorted([(np.mean(scores), -C) for C, scores in C_results.items()],
    #                    reverse=True)
    # print(C_results)
    # best_C = -C_results[0][1]

    # DEV: cached C values for the corpora we know
    # TODO: try things lower than 1e-3; might be necessary for GloVe sources
    if PIVOT == "mcrae":
        best_C = 1.0
    elif PIVOT == "wikigiga":
        best_C = 0.001
    elif PIVOT == "cc":
        best_C = 0.001

    reg = LogisticRegression(class_weight="balanced", fit_intercept=False,
                             C=best_C)
    cls = OneVsRestClassifier(reg, n_jobs=16)
    cls.fit(X, Y)

    preds = cls.predict(X)
    counts = Y.sum(axis=0)
    do_ignore = counts == 0
    ret_metrics = [metrics.f1_score(Y[:, i], preds[:, i]) if not ignore else None
                   for i, ignore in enumerate(do_ignore)]

    return zip(feature_names, counts, ret_metrics)


def get_values(input_file, c_string, value):
    concept_values = {}
    with open(input_file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row[value] == 'n/a':
                row[value] = 0
            concept_values[row[c_string]] = float(row[value])
    return concept_values


def plot_gaussian_contour(xs, ys, vars_xs, vars_ys):
    max_abs_x_var = np.abs(vars_xs).max()
    max_abs_y_var = np.abs(vars_xs).max()
    x_samp, y_samp = np.meshgrid(np.linspace(min(xs) - max_abs_x_var,
                                             max(xs) + max_abs_x_var,
                                             1000),
                                 np.linspace(min(ys) - max_abs_y_var,
                                             max(ys) + max_abs_y_var,
                                             1000))

    Cs = []
    for x, y, x_var, y_var in zip(xs, ys, vars_xs, vars_ys):
        gauss = mlab.bivariate_normal(x_samp, y_samp,
                                      mux=x, sigmax=x_var,
                                      muy=y, sigmay=y_var)
        C = plt.contour(x_samp, y_samp, gauss, alpha=0.8)
        Cs.append(C)

    return Cs


def produce_unified_domain_graph(vocab, features, feature_data, domain_concepts=None):
    domain_p1_means, domain_p1_vars = \
            domain_feat_freq.get_average(PEARSON1, 'Concept',
                                         'correlation', domain_concepts=domain_concepts)
    domain_p2_means, domain_p2_vars = \
            domain_feat_freq.get_average(PEARSON2, 'Concept',
                                        'correlation', domain_concepts=domain_concepts)
    assert domain_p1_means.keys() == domain_p2_means.keys()

    feature_map = {feature: weight for feature, _, weight in feature_data}

    # TODO: Arbitrary number
    all_domains = sorted([d for d in domain_concepts.keys()
        if len(domain_concepts[d]) > 7])

    xs, ys, zs, labels = [], [], [], []
    x_vars, y_vars, z_vars = [], [], []
    for domain in all_domains:
        weights = [feature_map[feature.name]
                   for feature in features.values()
                   for concept in domain_concepts[domain]
                   if concept in feature.concepts and feature.name
                   in feature_map]
        if not weights:
            continue

        xs.append(domain_p1_means[domain])
        ys.append(domain_p2_means[domain])
        zs.append(np.median(weights))
        x_vars.append(domain_p1_vars[domain])
        y_vars.append(domain_p2_vars[domain])
        z_vars.append(np.var(weights))
        labels.append(domain)

    # Resize Z values
    zs = np.array(zs)
    zs = (zs - zs.min()) / (zs.max() - zs.min())

    # Render Z axis using colors
    colormap = plt.get_cmap("cool")
    cs = colormap(zs)

    # # Jitter points
    # xs += np.random.randn(len(xs)) * 0.01
    # ys += np.random.randn(len(ys)) * 0.01

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel(PEARSON2_NAME)
    ax.set_zlabel("feature weight")
    ax.scatter(xs, ys, zs, c=cs)
    plt.show(fig)

    # Plot Pearson1 vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel(PEARSON2_NAME)
    ax.scatter(xs, ys, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], ys[i]), fontsize=15)

    plot_gaussian_contour(xs, ys, x_vars, y_vars)

    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-%s.svg" % (PEARSON1_NAME, PEARSON2_NAME))
    fig.savefig(fig_path)

    # Plot feature metric vs. Pearson1

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(xs, zs, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], zs[i]))

    plot_gaussian_contour(xs, zs, x_vars, z_vars)

    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.svg" % PEARSON1_NAME)
    fig.savefig(fig_path)

    # Plot feature metric vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON2_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(ys, zs, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (ys[i], zs[i]))

    plot_gaussian_contour(ys, zs, y_vars, z_vars)

    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.svg" % PEARSON2_NAME)
    fig.savefig(fig_path)


def analyze_domains(labels, ff_scores, concept_domains=None):
    if concept_domains is None:
        concept_domains = get_domains.get_concept_domains()
    x, y = [], []
    domain_feat_fit = {}
    for i, concept in enumerate(labels):
        for d in concept_domains[concept]:
            x.append(d)
            y.append(ff_scores[i])
            if d in domain_feat_fit:
                domain_feat_fit[d].append(ff_scores[i])
            else:
                domain_feat_fit[d] = [ff_scores[i]]
    print("average variance: ", np.mean([np.var(domain_feat_fit[d]) for d in domain_feat_fit]))
    # domain_averages = [np.mean(domain_feat_fit[d]) for d in domain_feat_fit]
    # domains_sorted = [x for (y,x) in sorted(zip(domain_averages,domain_feat_fit.keys()))]
    # print("domains sorted by average feature_fit: ", domains_sorted)
    sns.set_style("whitegrid")
    sns_plot = sns.swarmplot(x, y)
    sns_plot = sns.boxplot(x, y, showcaps=False,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':0})
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-domain.svg" % PIVOT)
    fig = sns_plot.get_figure()
    fig.savefig(fig_path)


def produce_unified_graph(vocab, features, feature_data, domain_concepts=None):
    concept_pearson1 = get_values(PEARSON1, "Concept", "correlation")
    concept_pearson2 = get_values(PEARSON2, 'Concept', 'correlation')
    assert concept_pearson1.keys() == concept_pearson2.keys()
    assert set(concept_pearson1.keys()) == set(vocab)

    #concepts_of_interest = get_domains.get_domain_concepts()[2]

    if domain_concepts is None:
        domain_concepts = get_domains.get_domain_concepts()
    domain_choices = [2, 12] #random.sample(domain_concepts.keys(), 5)
    domain_color_choices = ["DarkCyan", "Sienna"]#, "LightBlue", "SpringGreen", "MediumPurple"]
    interesting_domains = dict(zip(domain_choices,
        domain_color_choices))
    marker_choices = ["s", "d"]
    interesting_domains_markers = dict(zip(domain_choices, marker_choices))
    print("Color lengend", interesting_domains)

    feature_map = {feature: weight for feature, _, weight in feature_data}

    print("feature\tpvar\tpmean\twvar\twmean")
    for feature in features.values():
        this_features_concepts = set(vocab) & set(feature.concepts)
        if len(this_features_concepts) > 7:
            pearsons = [concept_pearson1[concept] for concept in this_features_concepts]
            wordnets = [concept_pearson2[concept] for concept in this_features_concepts]
            print("%s\t%f\t%f\t%f\t%f" % (feature.name, np.var(pearsons),
                np.mean(pearsons), np.var(wordnets), np.mean(wordnets)))

    xs, ys, zs, labels, colors, markers = [], [], [], [], [], []
    for concept in vocab:
        weights = [feature_map[feature.name]
                   for feature in features.values()
                   if concept in feature.concepts
                        and feature.name in feature_map]
        if not weights:
            continue
        colors.append("LightGray")
        markers.append("o")
        for d in interesting_domains:
            if concept in domain_concepts[d]:
                colors.pop()
                colors.append(interesting_domains[d])
                markers.pop()
                markers.append(interesting_domains_markers[d])
        xs.append(concept_pearson1[concept])
        ys.append(concept_pearson2[concept])
        zs.append(np.median(weights))
        labels.append(concept)

    concept_domains = {c: [d] for d, cs in domain_concepts.items() for c in cs}
    analyze_domains(labels, zs, concept_domains=concept_domains)

    # Resize Z values
    zs = np.array(zs)
    zs = (zs - zs.min()) / (zs.max() - zs.min())


    # Render Z axis using colors
    colormap = plt.get_cmap("cool")
    cs = colormap(zs)

    # Jitter points
    xs += np.random.randn(len(xs)) * 0.01
    ys += np.random.randn(len(ys)) * 0.01

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel(PEARSON2_NAME)
    ax.set_zlabel("feature weight")
    ax.scatter(xs, ys, zs, c=cs)
    plt.show()

    # Plot Pearson1 vs. Pearson2

    fig = plt.figure()
    #fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Pearson corr between " + SOURCE + " and " + PIVOT)
    ax.set_ylabel("Pearson corr between WordNet and " + PIVOT)
    # plot points of interest in front of other points
    for _m, _c, _x, _y in zip(markers, colors, xs, ys):
    	if _m == 'o':
    		ax.scatter(_x, _y, marker=_m, c=_c, alpha=0.8)
    for _m, _c, _x, _y in zip(markers, colors, xs, ys):
    	if _m != 'o':
    		ax.scatter(_x, _y, marker=_m, c=_c, alpha=0.8)
    # ax.scatter(xs, ys, c=_c, marker=_m, alpha=0.8) # c=cs
    for i, concept in enumerate(labels):
        if colors[i] != "LightGray":
            ax.annotate(concept, (xs[i], ys[i]))

    fig_path = os.path.join(GRAPH_DIR, "unified-%s-%s.svg" % (PEARSON1_NAME, PEARSON2_NAME))
    fig.savefig(fig_path)
    plt.close()

    # Plot feature metric vs. Pearson1

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(xs, zs, c=cs, alpha=0.8)

    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.svg" % PEARSON1_NAME)
    fig.savefig(fig_path)
    plt.close()

    # Plot feature metric vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON2_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(ys, zs, c=cs, alpha=0.8)

    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.svg" % PEARSON2_NAME)
    fig.savefig(fig_path)
    plt.close()


def cluster_metric_fn(x, y):
    x_weight, y_weight = x[0], y[0]
    x_emb, y_emb = x[1:], y[1:]

    emb_dist = distance.cosine(x_emb, y_emb)
    if np.isnan(x_weight) or np.isnan(y_weight):
        return emb_dist
    else:
        weight_dist = (x_weight - y_weight) ** 2
        return emb_dist + 100 * weight_dist


def try_cluster(k, X):
    from scipy.cluster.hierarchy import linkage
    return k, linkage(X, method="average", metric=cluster_metric_fn)


def do_cluster(vocab, features, feature_data):
    from get_domains import create_X, distance_siblings
    from scipy.cluster.hierarchy import linkage
    X, labels = create_X(vocab)
    assert set(labels) == set(vocab)

    # Add a new column to X: avg feature_fit metric
    feature_dict = {k: val for k, _, val in feature_data}
    concept_vals = defaultdict(list)
    for f_name, feature in features.items():
        for concept in feature.concepts:
            if concept in vocab and f_name in feature_dict:
                concept_vals[concept].append(feature_dict[f_name])
    concept_vals = {c: np.mean(vals) for c, vals in concept_vals.items()}

    mean_metric = [concept_vals.get(label, np.nan) for label in labels]
    X = np.append(np.array([mean_metric]).T, X, axis=1)

    # with futures.ProcessPoolExecutor(20) as executor:
    #     Z_futures = []
    #     for k in range(10, 101):
    #         Z_futures.append(executor.submit(try_cluster, k, X))

    # for Z_future in tqdm(futures.as_completed(Z_futures),
    #                      total=len(Z_futures), file=sys.stdout):
    #     k, Z_k = Z_future.result()
    #     sib_clusters = distance_siblings(Z_k, labels, k)
    #     domains = {d: concepts for d, concepts in enumerate(sib_clusters) if concepts}

    #     # domains = defaultdict(list)
    #     # for label, domain_idx in zip(labels, domain_maps):
    #     #     domains[domain_idx].append(label)

    #     # Compute average variance of metric within domain.
    #     n_singletons = len([d for d, cs in domains.items() if len(cs) == 1])
    #     domains = {d: np.var([concept_vals[c] for c in cs if c in concept_vals])
    #                for d, cs in domains.items()}
    #     d_metric = np.mean(list(domains.values()))
    #     print("%03i\t%5f\t%i" % (k, d_metric, n_singletons))

    Z = linkage(X, method="average", metric=cluster_metric_fn)
    sib_clusters = distance_siblings(Z, labels, 40)
    results = []
    for i, sib_cluster in enumerate(sib_clusters):
        if not sib_cluster: next
        weights = [concept_vals[c] for c in sib_cluster if c in concept_vals]
        results.append((i, np.mean(weights), np.var(weights), sib_cluster))

    results = sorted(results, key=lambda x: x[2])
    for idx, mean, var, items in results:
        print("%i\t%5f\t%5f\t%s" % (idx, mean, var, " ".join(items)))

    return {i: concepts for i, concepts in enumerate(sib_clusters)
            if len(concepts) > 0}


def main():
    features, concepts = load_features_concepts()
    vocab, embeddings = load_embeddings(concepts)
    word2idx = {w: i for i, w in enumerate(vocab)}

    feature_data = analyze_features(features, word2idx, embeddings)
    feature_data = sorted(filter(lambda f: f[2] is not None, feature_data),
                          key=lambda f: f[2])

    fcat_med = {}
    with open(OUTPUT, "w") as out:
        grouping_fns = {
            "br_label": lambda name: features[name].br_label,
            "first_word": lambda name: name.split("_")[0],
        }
        groups = {k: defaultdict(list) for k in grouping_fns}
        for name, n_entries, score in feature_data:
            out.write("%40s\t%25s\t%i\t%f\n" %
                        (name, features[name].br_label, n_entries, score))

            for grouping_fn_name, grouping_fn in grouping_fns.items():
                grouping_fn = grouping_fns[grouping_fn_name]
                group = grouping_fn(name)
                groups[grouping_fn_name][group].append((score, n_entries))

        for grouping_fn_name, groups_result in sorted(groups.items()):
            out.write("\n\nGrouping by %s:\n" % grouping_fn_name)
            summary = {}
            for name, data in groups_result.items():
                data = np.array(data)
                scores = data[:, 0]
                n_entries = data[:, 1]
                summary[name] = (len(data), np.mean(scores), np.percentile(scores, (0, 50, 100)),
                                np.mean(n_entries))
            summary = sorted(summary.items(), key=lambda x: x[1][2][1])

            out.write("%25s\tmu\tn\tmed\t\tmin\tmean\tmax\n" % "group")
            out.write(("=" * 100) + "\n")
            for label_group, (n, mean, pcts, n_concepts) in summary:
                out.write("%25s\t%.2f\t%3i\t%.5f\t\t%.5f\t%.5f\t%.5f\n"
                          % (label_group, n_concepts, n, pcts[1], pcts[0], mean, pcts[2]))
                fcat_med[label_group] = pcts[1]

    domain_concepts = do_cluster(vocab, features, feature_data)

    produce_unified_graph(vocab, features, feature_data, domain_concepts=domain_concepts)
    produce_unified_domain_graph(vocab, features, feature_data, domain_concepts=domain_concepts)


if __name__ == "__main__":
    main()
