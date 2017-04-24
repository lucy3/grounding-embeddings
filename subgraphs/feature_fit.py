import codecs
from collections import defaultdict, namedtuple
from multiprocessing import Process, Pool, sharedctypes
from pathlib import Path
from pprint import pprint
import csv
import os.path
import sys

import numpy as np
from numpy import ctypeslib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm, trange

import domain_feat_freq
import get_domains

# The "pivot" source is where we draw concept representations from. The
# resulting feature_fit metric represents how well these representations encode
# the relevant features. Each axis of the resulting graphs also involves the
# pivot source.
PIVOT = "cc"
if PIVOT == "mcrae":
    INPUT = "./all/mcrae_vectors.txt"
elif PIVOT == "wikigiga":
    INPUT = "../glove/glove.6B.300d.txt"
elif PIVOT == "cc":
    INPUT = "../glove/glove.840B.300d.txt"

FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
EMBEDDINGS = "./all/embeddings.%s.npy" % PIVOT

OUTPUT = "./all/feature_fit/mcrae_%s.txt" % PIVOT
PEARSON1_NAME = "mcrae_%s" % PIVOT if PIVOT != "mcrae" else "mcrae_wikigiga"
PEARSON1 = './all/pearson_corr/corr_%s.txt' % PEARSON1_NAME
PEARSON2_NAME = "%s_wordnetres" % PIVOT
PEARSON2 = './all/pearson_corr/corr_%s.txt' % PEARSON2_NAME
GRAPH_DIR = './all/feature_fit/%s' % PIVOT

Feature = namedtuple("Feature", ["name", "concepts", "wb_label", "wb_maj",
                                 "wb_min", "br_label", "disting"])


def load_embeddings(concepts):
    if Path(EMBEDDINGS).is_file():
        embeddings = np.load(EMBEDDINGS)

        assert Path(VOCAB).is_file()
        with open(VOCAB, "r") as vocab_f:
            vocab = [line.strip() for line in vocab_f]
        assert len(embeddings) == len(vocab), "%i %i" % (len(embeddings), len(vocab))
    else:
        vocab, embeddings = [], []
        with open(INPUT, "r") as glove_f:
            for line in glove_f:
                fields = line.strip().split()
                word = fields[0]
                if word in concepts:
                    vec = [float(x) for x in fields[1:]]
                    embeddings.append(vec)
                    vocab.append(word)

        voc_embeddings = sorted(zip(vocab, embeddings), key=lambda x: x[0])
        vocab = [x[0] for x in voc_embeddings]
        embeddings = [x[1] for x in voc_embeddings]

        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS, embeddings)

        with open(VOCAB, "w") as vocab_f:
            vocab_f.write("\n".join(vocab))

    return vocab, embeddings


def load_features_concepts():
    """
    Returns:
        features: string -> Feature
        concepts: set of strings
    """
    features = {}
    concepts = set()

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
    from collections import Counter
    from pprint import pprint
    print("# of features with particular number of associated concepts:")
    pprint(Counter(lengths))

    return features, concepts


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
    # samp_feats = np.random.choice(nonzero_features, replace=False, size=5)

    # # For each feature, retrieve concepts to sample as negative candidates in a
    # # ranking loss.
    # negsamp_concepts = [(1 - Y[:, f_idx]).nonzero()[0] for f_idx in samp_feats]

    # C_results = defaultdict(list)
    # for C_exp in trange(-3, 1, desc="C", file=sys.stdout):
    #     C = 10 ** C_exp
    #     reg = LogisticRegression(class_weight="balanced", fit_intercept=False,
    #                              C=C)
    #     cls = OneVsRestClassifier(reg, n_jobs=8)

    #     for f_idx, negsamps in tqdm(zip(samp_feats, negsamp_concepts),
    #                                 file=sys.stdout, desc="loocv outer"):
    #         c_idxs = Y[:, f_idx].nonzero()[0]
    #         for c_idx in tqdm(np.random.choice(c_idxs, replace=False, size=5),
    #                           file=sys.stdout, desc="loocv inner"):
    #             X_loo = np.concatenate([X[:c_idx], X[c_idx+1:]])
    #             Y_loo = np.concatenate([Y[:c_idx], Y[c_idx+1:]])

    #             cls_loo = clone(cls)
    #             cls_loo.fit(X_loo, Y_loo)

    #             # Draw negative samples for a ranking loss
    #             negsamps = np.random.choice(negsamps, replace=False, size=5)
    #             test = np.concatenate([X[c_idx:c_idx+1], X[negsamps]])
    #             pred_prob = cls_loo.predict_proba(X)[:, f_idx]
    #             score = pred_prob[0] - np.mean(pred_prob[1:])
    #             C_results[C].append(score)

    #     print(C, C_results[C])

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


def analyze_feature(feature, features, word2idx):
    """
    Compute metric for a given feature.

    Returns:
        feature_name:
        n_concepts: number of concepts which have this feature
        metric: goodness metric, where higher is better
    """

    # Get numpy array from shared data
    embeddings = ctypeslib.as_array(embeddings_shared)
    embeddings = embeddings.reshape(embedding_shape)

    # Fetch available embeddings.
    concepts = [concept for concept in features[feature].concepts
                if concept in word2idx]
    # embeddings = [embeddings[word2idx[concept]]
    #               for concept in concepts
    #               if concept in word2idx]
    if len(concepts) < 5:
        return

    X = embeddings
    y = np.zeros(len(word2idx))
    for concept in concepts:
        y[word2idx[concept]] = 1

    C_results = defaultdict(list)
    for C in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        for i in range(len(concepts)):
            X_loo = np.concatenate((X[:i], X[i+1:]))
            y_loo = np.concatenate((y[:i], y[i+1:]))

            reg_loo = linear_model.LogisticRegression(class_weight="balanced",
                                                      fit_intercept=False,
                                                      C=C)
            reg_loo.fit(X_loo, y_loo)

            prediction = reg_loo.score([X[i]], [y[i]])
            C_results[C].append(prediction)

    # Descending sort, first by accuracy and then by C (C is inverse strength;
    # prefer less regularization if equal perf)
    C_results = sorted([(np.mean(preds), -C) for C, preds in C_results.items()],
                        reverse=True)
    tqdm.write("%30s\t%s" % (feature,
                             "; ".join("%.2f %.0e" % (result[0], -result[1])
                                       for result in C_results)))
    best_C = -C_results[0][1]
    reg = linear_model.LogisticRegression(class_weight="balanced",
                                          fit_intercept=False, C=best_C)
    reg.fit(X, y)
    metric = reg.score(X, y)

    # embeddings = np.array(embeddings)
    # mean_v = embeddings.mean(axis=0)
    # dists = embeddings @ mean_v
    # dists /= np.linalg.norm(embeddings, axis=1)
    # dists /= np.linalg.norm(mean_v)
    # metric = dists.mean()

    return feature, len(concepts), metric


def produce_domain_graphs(fcat_med):
    domain_pearson1 = domain_feat_freq.get_average(PEARSON1, 'Concept',
        'correlation')
    domain_pearson2 = domain_feat_freq.get_average(PEARSON2, 'Concept',
        'correlation')
    domain_matrix, domains, fcat_list = domain_feat_freq.get_feat_freqs(weights=fcat_med)
    domain_feat_freq.render_graphs(GRAPH_DIR, domain_pearson1, domain_pearson2,
        domains, domain_matrix, fcat_list)


def get_values(input_file, c_string, value):
    concept_values = {}
    with open(input_file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row[value] == 'n/a':
                row[value] = 0
            concept_values[row[c_string]] = float(row[value])
    return concept_values


def get_fcat_conc_freqs(vocab, weights=None):
    '''
    @inputs:
    - vocab: sorted list of concepts
    - weights: {fcat: med}
    '''
    feature_cats = set()
    concept_feats = {c: [] for c in vocab}
    with open(FEATURES, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row["Concept"] in vocab:
                concept_feats[row["Concept"]].append((row["BR_Label"], row["Prod_Freq"]))
                feature_cats.add(row["BR_Label"])
    fcat_list = sorted(list(feature_cats))

    concept_matrix = np.zeros((len(vocab), len(fcat_list))) # rows: domains, columns: feature categories
    for i in range(len(vocab)):
        feats = concept_feats[vocab[i]] # list of tuples (feature category, production frequency)
        for f in feats:
            if weights and f[0] != "smell":
                concept_matrix[i][fcat_list.index(f[0])] += weights[f[0]]*int(f[1])
            else:
                concept_matrix[i][fcat_list.index(f[0])] += int(f[1])
    concept_totals = np.sum(concept_matrix, axis=1)
    concept_matrix = concept_matrix/concept_totals[:,None]

    return(concept_matrix, fcat_list)


def produce_concept_graphs(fcat_med):
    """
    TODO: If we still want to use this function,
    we should be sure to label the axes
    """
    concept_pearson1 = get_values(PEARSON1, 'Concept', 'correlation')
    concept_pearson2 = get_values(PEARSON2, 'Concept', 'correlation')
    assert concept_pearson1.keys() == concept_pearson2.keys()
    vocab = sorted(concept_pearson1.keys())
    concept_matrix, fcat_list = get_fcat_conc_freqs(vocab, weights=fcat_med)
    print(concept_matrix)

    xs = [concept_pearson1[concept] for concept in vocab]
    ys = [concept_pearson2[concept] for concept in vocab]
    concept_matrix = (concept_matrix - concept_matrix.min(axis=0)) / (concept_matrix.max(axis=0)
        - concept_matrix.min(axis=0))
    colormap = plt.get_cmap("cool")

    # For each feature category, produce a scatter plot and use feature
    # category metrics to color the points (as a sort of third dimension).
    for j, fcat in enumerate(fcat_list):
        print(fcat)
        fig = plt.figure()
        fig.suptitle(fcat+"-08-60-concepts-perc")

        ax = fig.add_subplot(111)
        c = colormap([concept_matrix[i, j] for i, concept in enumerate(vocab)])
        ax.scatter(xs, ys, [], c)

        fig_path = os.path.join(GRAPH_DIR, fcat)
        fig.savefig(fig_path + '-08-60-concepts-perc')

def produce_unified_domain_graph(vocab, features, feature_data):
    domain_pearson1 = domain_feat_freq.get_average(PEARSON1, 'Concept',
        'correlation')
    domain_pearson2 = domain_feat_freq.get_average(PEARSON2, 'Concept',
        'correlation')
    assert domain_pearson1.keys() == domain_pearson2.keys()

    feature_map = {feature: weight for feature, _, weight in feature_data}

    domain_concepts = get_domains.get_domain_concepts()
    all_domains = sorted([d for d in domain_concepts.keys()
        if len(domain_concepts[d]) > 7])

    xs, ys, zs, labels = [], [], [], []
    for domain in all_domains:
        weights = [feature_map[feature.name]
                   for feature in features.values()
                   for concept in domain_concepts[domain]
                   if concept in feature.concepts and feature.name
                   in feature_map]
        if not weights:
            continue

        xs.append(domain_pearson1[domain])
        ys.append(domain_pearson2[domain])
        zs.append(np.median(weights))
        labels.append(domain)

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
    plt.show(fig)

    # HACK: trying to make this approximately normal so that I can easily see
    # the differences between points. shave off high outliers.
    zs = np.clip(zs, 0, 0.7)
    zs = zs / zs.max()

    # Plot Pearson1 vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel(PEARSON2_NAME)
    ax.scatter(xs, ys, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], ys[i]))

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

    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.png" % PEARSON2_NAME)
    fig.savefig(fig_path)

def analyze_domains(labels, ff_scores):
    concept_domains = get_domains.get_concept_domains()
    x, y = [], []
    for i, concept in enumerate(labels):
        for d in concept_domains[concept]:
            x.append(d)
            y.append(ff_scores[i])
    fig = plt.figure()
    fig.suptitle("domain graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel("domains")
    ax.set_ylabel("feature fit")
    ax.scatter(x, y)
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-domain.png" % PIVOT)
    fig.savefig(fig_path)


def produce_unified_graph(vocab, features, feature_data):
    concept_pearson1 = get_values(PEARSON1, "Concept", "correlation")
    concept_pearson2 = get_values(PEARSON2, 'Concept', 'correlation')
    assert concept_pearson1.keys() == concept_pearson2.keys()
    assert set(concept_pearson1.keys()) == set(vocab)

    concepts_of_interest = get_domains.get_domain_concepts()[3]

    feature_map = {feature: weight for feature, _, weight in feature_data}

    print("feature\tpvar\tpmean\twvar\twmean")
    for feature in features.values():
        this_features_concepts = set(vocab) & set(feature.concepts)
        if len(this_features_concepts) > 7:
            pearsons = [concept_pearson1[concept] for concept in this_features_concepts]
            wordnets = [concept_pearson2[concept] for concept in this_features_concepts]
            print("%s\t%f\t%f\t%f\t%f" % (feature.name, np.var(pearsons),
                np.mean(pearsons), np.var(wordnets), np.mean(wordnets)))

    xs, ys, zs, labels = [], [], [], []
    for concept in vocab:
        weights = [feature_map[feature.name]
                   for feature in features.values()
                   if concept in feature.concepts
                        and feature.name in feature_map]
        if not weights:
            continue

        xs.append(concept_pearson1[concept])
        ys.append(concept_pearson2[concept])
        zs.append(np.median(weights))
        labels.append(concept)

    analyze_domains(labels, zs)

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

    # HACK: trying to make this approximately normal so that I can easily see
    # the differences between points. shave off high outliers.
    zs = np.clip(zs, 0, 0.7)
    zs = zs / zs.max()

    # Plot Pearson1 vs. Pearson2

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON1_NAME)
    ax.set_ylabel(PEARSON2_NAME)
    ax.scatter(xs, ys, c=cs, alpha=0.8)
    for i, concept in enumerate(labels):
        if concept in concepts_of_interest:
            ax.annotate(concept, (xs[i], ys[i]))

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

    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.png" % PEARSON2_NAME)
    fig.savefig(fig_path)
    plt.close()


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

    produce_unified_graph(vocab, features, feature_data)
    produce_unified_domain_graph(vocab, features, feature_data)

    #produce_domain_graphs(fcat_med) # this calls functions in domain_feat_freq.py
    #produce_concept_graphs(fcat_med) # this calls functions in here


if __name__ == "__main__":
    main()
