import codecs
from collections import defaultdict, namedtuple, Counter
from concurrent import futures
from functools import partial
import itertools
from pathlib import Path
from pprint import pprint
import random
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
from scipy import stats

import domain_feat_freq
import get_domains
import random

# The "pivot" source is where we draw concept representations from. The
# resulting feature_fit metric represents how well these representations encode
# the relevant features. Each axis of the resulting graphs also involves the
# pivot source.
PIVOT = "cc"
if PIVOT == "mcrae":
    INPUT = "./all/mcrae_vectors.txt"
elif PIVOT == "cslb":
    INPUT = "./all/cslb_vectors.txt"
elif PIVOT == "wikigiga":
    INPUT = "../glove/glove.6B.300d.txt"
elif PIVOT == "cc":
    INPUT = "../glove/glove.840B.300d.txt"
elif PIVOT == "word2vec":
    INPUT = "../word2vec/GoogleNews-vectors-negative300.bin"

SOURCE = "mcrae"
if SOURCE == "mcrae":
    FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
else:
    FEATURES = "../cslb/norms.dat"
VOCAB = "./all/vocab_%s.txt" % SOURCE
EMBEDDINGS = "./all/embeddings.%s.%s.npy" % (SOURCE, PIVOT)

# Auxiliary input paths
PEARSON1_NAME = "%s_%s" % (SOURCE, PIVOT) if PIVOT != SOURCE else "%s_wikigiga" % SOURCE
PEARSON1 = './all/pearson_corr/%s/corr_%s.txt' % (SOURCE, PEARSON1_NAME)
PEARSON2_NAME = "wordnetres_%s" % PIVOT
PEARSON2 = './all/pearson_corr/%s/corr_%s.txt' % (SOURCE, PEARSON2_NAME)
GRAPH_DIR = './all/feature_fit/%s/%s' % (SOURCE, PIVOT)

# Output paths
OUT_DIR = "./all/feature_fit/%s/%s" % (SOURCE, PIVOT)
CV_OUTPUT = "%s/Cs.txt" % OUT_DIR
FF_OUTPUT = "%s/features.txt" % OUT_DIR
GROUP_OUTPUT = "%s/groups.txt" % OUT_DIR
CLUSTER_OUTPUT = "%s/clusters.txt" % OUT_DIR
LOG = "%s/log.txt" % OUT_DIR

if PIVOT == "wikigiga":
    PIVOT_FORMAL = "Wikipedia+Gigaword"
elif PIVOT == "cc":
    PIVOT_FORMAL = "Common Crawl"

if SOURCE == "cslb":
    SOURCE_FORMAL = "CSLB"
elif SOURCE == "mcrae":
    SOURCE_FORMAL = "McRae"

Feature = namedtuple("Feature", ["name", "concepts", "wb_label", "wb_maj",
                                 "wb_min", "br_label", "disting"])


def load_embeddings(concepts):
    assert Path(VOCAB).is_file()
    with open(VOCAB, "r") as vocab_f:
        vocab = [line.strip() for line in vocab_f]

    if Path(EMBEDDINGS).is_file():
        embeddings = np.load(EMBEDDINGS)
        assert len(embeddings) == len(vocab), "%i %i" % (len(embeddings), len(vocab))
    elif PIVOT == "word2vec":
        from gensim.models.keyedvectors import KeyedVectors
        model = KeyedVectors.load_word2vec_format(INPUT, binary=True)

        embeddings = []
        for concept in vocab:
            w2v_concept = concept
            if concept == "axe": w2v_concept = "ax"
            elif concept == "armour": w2v_concept = "armor"
            embeddings.append(model[w2v_concept])

        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS, embeddings)
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
            reader = csv.DictReader(features_f, delimiter='\t')
            for row in reader:
                concept_name = row["Concept"]
                feature_name = row["Feature"]
                if feature_name not in features:
                    br_label = row["BR_Label"]
                    # NB: override feature category for beh-, inbeh-
                    if "beh_-" in feature_name:
                        br_label = "function"

                    features[feature_name] = Feature(feature_name, set(),
                            row["WB_Label"], row["WB_Maj"], row["WB_Min"], br_label,
                            row["Disting"])
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


def load_loocv(features, *loocv_args):
    """
    Load (either by reading from a file, or by computing) LOOCV-validated
    regularization coefficients for each feature in `features`.
    """

    loocv_path = Path(CV_OUTPUT)
    if loocv_path.exists():
        with loocv_path.open("r") as loocv_f:
            Cs = {}
            for line in loocv_f:
                feature, C = line.strip().split("\t")
                Cs[feature] = float(C)
    else:
        Cs = loocv_features(features, *loocv_args)
        with loocv_path.open("w") as loocv_f:
            for feature, C in Cs.items():
                loocv_f.write("%s\t%f\n" % (feature, C))

    assert set(features) == set(Cs.keys())

    return Cs


def loocv_features(features, X, Y, clf_base):
    """
    Run LOOCV for all features.
    """

    # Share as global variables
    loocv_features.X = X
    loocv_features.Y = Y

    # Run LOOCV of cross product on features and C choices
    C_futures = {}
    C_results = defaultdict(dict)
    with futures.ThreadPoolExecutor(20) as executor:
        for f_idx, _ in enumerate(features):
            C_futures[f_idx] = loocv_feature_outer(executor, clf_base, f_idx)

        all_futures = list(itertools.chain.from_iterable(C_futures.values()))
        for future in tqdm(futures.as_completed(all_futures), total=len(all_futures),
                           file=sys.stdout):
            f_idx, C, scores = future.result()
            C_results[f_idx][C] = scores

    # Find best C for each feature
    final_results = {}
    best_Cs = Counter()
    for f_idx, Cs in C_results.items():
        best_C = max(Cs, key=lambda C: np.median(Cs[C]))
        best_Cs[best_C] += 1
        final_results[features[f_idx]] = best_C

    print(best_Cs)
    return final_results


def loocv_feature_outer(pool, clf_base, f_idx):
    Cs = [10 ** exp for exp in range(-4, 4)]
    Cs += [5 * (10 ** exp) for exp in range(-4, 2)]
    Cs += [75, 25]

    return [pool.submit(loocv_feature, C, f_idx, clf_base(C=C))
            for C in Cs]


def loocv_feature(C, f_idx, clf):
    """
    Evaluate LOOCV regression on a given feature with a given classifier
    instance.
    """

    # Retrieve shared vars.
    X = loocv_features.X
    y = loocv_features.Y[:, f_idx]

    scores = []

    # Find all concepts which (1) do or (2) do not have this feature
    c_idxs = y.nonzero()[0]
    c_not_idxs = (1 - y).nonzero()[0]

    # Leave-one-out.
    for c_idx in c_idxs:
        X_loo = np.concatenate([X[:c_idx], X[c_idx+1:]])
        y_loo = np.concatenate([y[:c_idx], y[c_idx+1:]])

        clf_loo = clone(clf)
        clf_loo.fit(X_loo, y_loo)

        # Draw negative samples for a ranking loss
        test = np.concatenate([X[c_idx:c_idx+1], X[c_not_idxs]])
        pred_prob = clf_loo.predict_proba(test)[:, 1]

        score = np.log(pred_prob[0]) + np.mean(np.log(1 - pred_prob[1:]))
        scores.append(score)

    return f_idx, C, scores


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

    usable_features = []
    feature_concepts = {}
    for feature_name in sorted(features.keys()):
        feature = features[feature_name]
        concepts = [word2idx[c] for c in feature.concepts if c in word2idx]
        if len(concepts) < 5:
            continue

        usable_features.append(feature.name)
        feature_concepts[feature.name] = concepts

    # Prepare for a multi-label logistic regression.
    X = embeddings
    Y = np.zeros((len(word2idx), len(usable_features)))

    for f_idx, f_name in enumerate(usable_features):
        for c_idx in feature_concepts[f_name]:
            Y[c_idx, f_idx] = 1

    ############
    # Load LOOCV-validated regression models for each feature.

    clf_base = partial(LogisticRegression, class_weight="balanced",
                       fit_intercept=False)

    Cs = load_loocv(usable_features, X, Y, clf_base)
    clfs = {}
    for f_idx, f_name in tqdm(enumerate(usable_features),
                              total=len(usable_features),
                              desc="Training feature classifiers"):
        clfs[f_idx] = clf_base(C=Cs[f_name])
        clfs[f_idx].fit(X, Y[:, f_idx])

    counts = Y.sum(axis=0)
    ret_metrics = [metrics.f1_score(Y[:, f_idx],
                                    clfs[f_idx].predict(X))
                   for f_idx, _ in enumerate(usable_features)]

    # HACK: for dev
    usable_features = usable_features[:len(ret_metrics)]
    counts = counts[:len(ret_metrics)]

    return zip(usable_features, counts, ret_metrics)


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
    x_samp, y_samp = np.meshgrid(np.linspace(min(0, min(xs) - max_abs_x_var),
                                             max(0.8, max(xs) + max_abs_x_var),
                                             1000),
                                 np.linspace(min(0, min(ys) - max_abs_y_var),
                                             max(0.8, max(ys) + max_abs_y_var),
                                             1000))

    Cs = []
    for x, y, x_var, y_var in zip(xs, ys, vars_xs, vars_ys):
        x_stddev = np.sqrt(x_var)
        y_stddev = np.sqrt(y_var)

        gauss = mlab.bivariate_normal(x_samp, y_samp,
                                      mux=x, sigmax=x_stddev,
                                      muy=y, sigmay=y_stddev)

        # Draw level curves at 1 and 2 stddev away
        x_level = x + np.array([0.5]) * x_stddev
        y_level = y + np.array([0.5]) * y_stddev
        levels = mlab.bivariate_normal(x_level, y_level,
                                       mux=x, sigmax=x_stddev,
                                       muy=y, sigmay=y_stddev)
        levels = list(sorted(levels))

        C = plt.contour(x_samp, y_samp, gauss, levels=levels,
                        colors="black", alpha=0.8)
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
    #fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Pearson corr between " + SOURCE_FORMAL + " and " + PIVOT_FORMAL)
    ax.set_ylabel("Pearson corr between WordNet and " + PIVOT_FORMAL)
    ax.scatter(xs, ys, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], ys[i]), fontsize=20)

    plot_gaussian_contour(xs, ys, x_vars, y_vars)
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-%s.png" % (PEARSON1_NAME, PEARSON2_NAME))
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
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.png" % PEARSON1_NAME)
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
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.png" % PEARSON2_NAME)
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
    fig, ax = plt.subplots(figsize=(12, 4.8))
    sns_plot = sns.swarmplot(x, y, ax=ax)
    sns_plot = sns.boxplot(x, y, showcaps=False,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':0}, ax=ax)
    sns_plot.set(xlabel='domain ID', ylabel='feature fit score')
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-domain.png" % PIVOT)
    plt.tight_layout()
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

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    print("Pearson vs Pearson")
    print("slope", slope, "r", r_value)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, zs)
    print("Pearson vs Feature Fit")
    print("slope", slope, "r", r_value)

    # Plot Pearson1 vs. Pearson2

    fig = plt.figure()
    #fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Pearson corr between " + SOURCE_FORMAL + " and " + PIVOT_FORMAL)
    ax.set_ylabel("Pearson corr between WordNet and " + PIVOT_FORMAL)
    ax.scatter(xs, ys, c=cs)
    # # plot points of interest in front of other points
    # for _m, _c, _x, _y in zip(markers, colors, xs, ys):
    #   if _m == 'o':
    #       ax.scatter(_x, _y, marker=_m, c=_c, alpha=0.8)
    # for _m, _c, _x, _y in zip(markers, colors, xs, ys):
    #   if _m != 'o':
    #       ax.scatter(_x, _y, marker=_m, c=_c, alpha=0.8)
    # # ax.scatter(xs, ys, c=_c, marker=_m, alpha=0.8) # c=cs
    # for i, concept in enumerate(labels):
    #     if colors[i] != "LightGray":
    #         ax.annotate(concept, (xs[i], ys[i]))

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified-%s-%s.png" % (PEARSON1_NAME, PEARSON2_NAME))
    fig.savefig(fig_path)
    plt.close()

    # Plot feature metric vs. Pearson1

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(xs, zs, c=cs, alpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.png" % PEARSON1_NAME)
    fig.savefig(fig_path)
    plt.close()

    # Plot feature metric vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON2_NAME)
    ax.set_ylabel("feature_fit")
    ax.scatter(ys, zs, c=cs, alpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.png" % PEARSON2_NAME)
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
    for sib_cluster in sib_clusters:
        if not sib_cluster: next
        weights = [concept_vals[c] for c in sib_cluster if c in concept_vals]
        results.append((np.mean(weights), np.var(weights), sib_cluster))

    results = sorted(results, key=lambda x: x[0])
    results = [(i,) + result for i, result in enumerate(results)]
    with open(CLUSTER_OUTPUT, "w") as cluster_f:
        for idx, mean, var, items in results:
            out_str = "%i\t%5f\t%5f\t%s\n" % (idx, mean, var, " ".join(items))
            cluster_f.write(out_str)

    return {i: concepts for i, _, _, concepts in results
            if len(concepts) > 0}


def produce_feature_fit_bars(feature_groups, features_per_category=4):
    """
    Produce feature_fit bar charts with random samples from feature categories.
    """

    # HACK: fixed sort
    group_names = ["visual perceptual", "encyclopaedic", "other perceptual", "functional", "taxonomic"]

    fig, axes = plt.subplots(ncols=len(feature_groups), sharey=True, figsize=(15, 5))
    if not isinstance(axes, (tuple, list)):
        axes = [axes]
    for group_name, ax in zip(group_names, axes):
        group = sorted(feature_groups[group_name], key=lambda x: x[1])

        # Randomly sample low and high features
        low = random.sample(group[:int(len(group)/4)], int(features_per_category/2))
        high = random.sample(group[int(len(group)/4*3):], int(features_per_category/2))
        features_i = low + high

        data = sorted([(name, score) for name, score, _ in features_i],
                      key=lambda x: x[1])
        bp = sns.barplot(x=[name for name, _ in data], y=[score for _, score in data], ax=ax)
        bp.set_xticklabels([name for name, _ in data], rotation=45)
        ax.set_title(group_name)

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "feature_fit.png")
    fig.savefig(fig_path)


def do_bootstrap_test(feature_groups, pop1, pop2, n_bootstrap_samples=10000,
                      percentiles=(5, 95)):
    """
    Do a percentile bootstrap test on the difference of medians among features
    from different groups of categories.

    The hypothesis here is:

        median(features from pop2) - median(features from pop1) > 0
    """

    # Concatenate all features from pop1, pop2 into flat groups.
    # TODO: maybe stratified sampling would be better?
    pop1_features = list(itertools.chain.from_iterable(
            feature_groups[group] for group in pop1))
    pop2_features = list(itertools.chain.from_iterable(
            feature_groups[group] for group in pop2))

    # Convert these into easy-to-use NP arrays..
    pop1_features = np.array([score for _, score, _ in pop1_features])
    pop2_features = np.array([score for _, score, _ in pop2_features])

    tqdm.write("======= BOOTSTRAP ========")
    diffs = []
    for _ in trange(n_bootstrap_samples, desc="bootstrap"):
        pop1_samples = np.random.choice(pop1_features, size=len(pop1_features),
                                        replace=True)
        pop2_samples = np.random.choice(pop2_features, size=len(pop2_features),
                                        replace=True)

        diff = np.median(pop2_samples) - np.median(pop1_samples)
        diffs.append(diff)

    result = np.percentile(diffs, percentiles)

    tqdm.write("pop1: " + (" ".join(pop1)))
    tqdm.write("pop2: " + (" ".join(pop2)))
    tqdm.write("%i%% CI on (pop2 - pop1): %s" % (percentiles[1], result))
    tqdm.write("==========================")
    return result

def swarm_feature_cats(feature_groups, fcat_mean):
    fcats_sorted = sorted(feature_groups.keys(), key=lambda k: fcat_mean[k])
    x, y = [], []
    for fg in fcats_sorted:
        for _, score, _ in feature_groups[fg]:
            if fg == "visual-form_and_surface":
                x.append("visual-\nform_and_surface")
            elif fg == "taste" or fg == 'smell' or fg == 'sound':
                x.append("taste/smell/\nsound")
            else:
                x.append(fg)
            y.append(score)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns_plot = sns.swarmplot(x, y, ax=ax)
    sns_plot = sns.boxplot(x, y, showcaps=False,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':0}, ax=ax)
    sns_plot.set(xlabel='feature category', ylabel='feature fit score')
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-%s-category.png" % (SOURCE, PIVOT))
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(fig_path)


def main():
    features, concepts = load_features_concepts()
    vocab, embeddings = load_embeddings(concepts)
    word2idx = {w: i for i, w in enumerate(vocab)}

    feature_data = analyze_features(features, word2idx, embeddings)
    feature_data = sorted(feature_data, key=lambda f: f[2])

    fcat_mean = {}

    grouping_fns = {
        "br_label": lambda name: features[name].br_label,
        "first_word": lambda name: name.split("_")[0],
    }
    groups = {k: defaultdict(list) for k in grouping_fns}

    # Output raw feature data and group features
    with open(FF_OUTPUT, "w") as ff_out:
        for name, n_entries, score in feature_data:
            ff_out.write("%s\t%s\t%i\t%f\n" %
                         (name, features[name].br_label, n_entries, score))

            for grouping_fn_name, grouping_fn in grouping_fns.items():
                grouping_fn = grouping_fns[grouping_fn_name]
                group = grouping_fn(name)
                groups[grouping_fn_name][group].append((name, score, n_entries))

    # Output grouped feature information
    with open(GROUP_OUTPUT, "w") as group_out:
        for grouping_fn_name, groups_result in sorted(groups.items()):
            group_out.write("\n\nGrouping by %s:\n" % grouping_fn_name)
            summary = {}
            for group_name, group_data in groups_result.items():
                scores = np.array([group_data_i[1] for group_data_i in group_data])
                n_entries = np.array([group_data_i[2] for group_data_i in group_data])
                summary[group_name] = (len(group_data), np.mean(scores),
                                       np.percentile(scores, (0, 50, 100)),
                                       np.mean(n_entries))
            summary = sorted(summary.items(), key=lambda x: x[1][2][1])

            group_out.write("%25s\tmu\tn\tmed\t\tmin\tmean\tmax\n" % "group")
            group_out.write(("=" * 100) + "\n")
            for label_group, (n, mean, pcts, n_concepts) in summary:
                group_out.write("%25s\t%.2f\t%3i\t%.5f\t\t%.5f\t%.5f\t%.5f\n"
                                % (label_group, n_concepts, n, pcts[1], pcts[0],
                                   mean, pcts[2]))
                fcat_mean[label_group] = mean

    #produce_feature_fit_bars(groups["br_label"])
    if SOURCE == "cslb":
        do_bootstrap_test(groups["br_label"],
                          ["visual perceptual", "other perceptual"],
                          ["taxonomic", "function"])
    elif SOURCE == "mcrae":
        do_bootstrap_test(groups["br_label"],
                          ["visual-motion", "visual-form_and_surface", "visual-colour",
                           "sound", "tactile", "smell", "taste"],
                          ["function", "taxonomic"])

    swarm_feature_cats(groups["br_label"], fcat_mean)

    domain_concepts = do_cluster(vocab, features, feature_data)
    produce_unified_graph(vocab, features, feature_data, domain_concepts=domain_concepts)
    produce_unified_domain_graph(vocab, features, feature_data, domain_concepts=domain_concepts)


if __name__ == "__main__":
    main()
