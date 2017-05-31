from collections import defaultdict, namedtuple, Counter
from concurrent import futures
from functools import partial
import itertools
from pathlib import Path
import pickle
from pprint import pprint
import random
import csv
import os.path
import sys
import string

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from sklearn import metrics
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, trange
from scipy import stats

import domain_feat_freq
import get_domains
from util import get_map_from_tsv


# The "pivot" source is where we draw concept representations from. The
# resulting feature_fit metric represents how well these representations encode
# the relevant features. Each axis of the resulting graphs also involves the
# pivot source.
PIVOT = "wikigiga"
INPUT_FVOCAB = None
if PIVOT == "mcrae":
    INPUT = "./all/mcrae_vectors.txt"
elif PIVOT == "cslb":
    INPUT = "./all/cslb_vectors.txt"
elif PIVOT == "wikigiga":
    INPUT = "../glove/glove.6B.300d.w2v.txt"
    INPUT_FVOCAB = "/john0/scr1/jgauthie/vocab.txt"
    MIN_WORD_COUNT = 300
elif PIVOT == "cc":
    INPUT = "../glove/glove.840B.300d.w2v.txt"
    INPUT_FVOCAB = "/john0/scr1/jgauthie/vocab.txt"
    MIN_WORD_COUNT = 300
elif PIVOT == "word2vec":
    INPUT = "../word2vec/GoogleNews-vectors-negative300.bin"
    MIN_WORD_COUNT = 300 * 30
elif PIVOT == "mygiga":
    INPUT = "/john0/scr1/jgauthie/vectors.en.w2v.txt"
    INPUT_FVOCAB = "/john0/scr1/jgauthie/vocab.txt"
    MIN_WORD_COUNT = 300

SOURCE = "cslb"
if SOURCE == "mcrae":
    FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
elif SOURCE == "cslb" or SOURCE == "cslb_cutoff":
    FEATURES = "../cslb/norms.dat"
VOCAB = "./all/vocab_%s.txt" % SOURCE
EMBEDDINGS = "./all/embeddings.%s.%s.npy" % (SOURCE, PIVOT)
ALL_EMBEDDINGS = "./all/embeddings_all.%s.%s.bin" % (SOURCE, PIVOT)

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
FF_ALL_OUTPUT = "%s/features_concepts.txt" % OUT_DIR
GROUP_OUTPUT = "%s/groups.txt" % OUT_DIR
CLUSTER_OUTPUT = "%s/clusters.txt" % OUT_DIR
CONCEPT_OUTPUT = "%s/concepts.txt" % OUT_DIR
CLASSIFIER_OUTPUT = "%s/classifiers.pkl" % OUT_DIR
CLASSIFIER_NEIGHBOR_OUTPUT = "%s/classifier_neighbors.txt" % OUT_DIR
LOG = "%s/log.txt" % OUT_DIR

if PIVOT == "wikigiga":
    PIVOT_FORMAL = "GloVe-WG"
elif PIVOT == "cc":
    PIVOT_FORMAL = "GloVe-CC"
elif PIVOT == "word2vec":
    PIVOT_FORMAL = "word2vec"

if SOURCE == "cslb":
    SOURCE_FORMAL = "CSLB"
elif SOURCE == "mcrae":
    SOURCE_FORMAL = "McRae"

SAVEFIG_KWARGS = {"transparent": True, "bbox_inches": "tight", "pad_inches": 0}


Feature = namedtuple("Feature", ["name", "concepts", "wb_label", "wb_maj",
                                 "wb_min", "br_label", "disting"])
AnalyzeResult = namedtuple("AnalyzeResult",
                           ["feature", "n_concepts", "clf", "metric"])


def load_all_embeddings():
    if Path(ALL_EMBEDDINGS).is_file():
        embeddings = KeyedVectors.load(ALL_EMBEDDINGS)
    elif PIVOT == "word2vec":
        embeddings = KeyedVectors.load_word2vec_format(INPUT, binary=True)
    else:
        # GloVe format
        embeddings = KeyedVectors.load_word2vec_format(INPUT, binary=False,
                                                       fvocab=INPUT_FVOCAB)

    if not Path(ALL_EMBEDDINGS).is_file():
        embeddings.save(ALL_EMBEDDINGS)

    return embeddings


def load_filtered_embeddings(concepts, all_embeddings):
    assert Path(VOCAB).is_file()
    with open(VOCAB, "r") as vocab_f:
        vocab = [line.strip() for line in vocab_f]

    if Path(EMBEDDINGS).is_file():
        embeddings = np.load(EMBEDDINGS)
        assert len(embeddings) == len(vocab), "%i %i" % (len(embeddings), len(vocab))
    else:
        embeddings = []
        for concept in vocab:
            w2v_concept = concept
            if concept == "axe": w2v_concept = "ax"
            elif concept == "armour": w2v_concept = "armor"
            elif concept == "doughnut": w2v_concept = "donut"
            elif concept == "pyjamas": w2v_concept = "pajamas"
            elif concept == "aeroplane": w2v_concept = "airplane"
            elif concept == "tyre": w2v_concept = "tire"
            elif concept == "plough": w2v_concept = "plow"
            elif concept == "catalogue": w2v_concept = "catalog"
            elif concept == "whisky": w2v_concept = "whiskey"
            embeddings.append(all_embeddings[w2v_concept])

    embeddings = np.asarray(embeddings)
    if not Path(EMBEDDINGS).exists():
        np.save(embeddings, EMBEDDINGS)

    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

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

    elif SOURCE.startswith("cslb"):
        with open(FEATURES, "r") as features_f:
            reader = csv.DictReader(features_f, delimiter='\t')
            for row in reader:
                concept_name = row["concept"]
                feature_name = "_".join(row["feature"].split())

                if SOURCE == "cslb_cutoff":
                    pf = float(row["pf"])
                    if pf < 5: continue

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
    Cs = [10 ** exp for exp in range(-4, 3)]
    Cs += [5 * (10 ** exp) for exp in range(-4, 1)]

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

    # Find all concepts which (1) do or (2) do not have this feature
    c_idxs = y.nonzero()[0]
    c_not_idxs = (1 - y).nonzero()[0]

    scores = []
    def eval_clf(clf, X_test, y_test):
        pred_prob = clf.predict_proba(X_test)[:, 1]

        pos_probs = np.log(pred_prob[y_test == 1])
        neg_probs = np.log(1 - pred_prob[y_test == 0])

        return np.mean(pos_probs) + np.mean(neg_probs)

    if len(c_idxs) > 15:
        # Run 10-fold CV.
        skf = StratifiedKFold(n_splits=10)
        for train_index, test_index in skf.split(X, y):
            clf_kf = clone(clf)
            clf_kf.fit(X[train_index], y[train_index])
            scores.append(eval_clf(clf_kf, X[test_index], y[test_index]))
    else:
        # Run leave-one-out.
        for c_idx in c_idxs:
            X_loo = np.concatenate([X[:c_idx], X[c_idx+1:]])
            y_loo = np.concatenate([y[:c_idx], y[c_idx+1:]])

            clf_loo = clone(clf)
            clf_loo.fit(X_loo, y_loo)

            X_test = np.concatenate([X[c_idx:c_idx+1], X[c_not_idxs]])
            y_test = np.concatenate([[1], np.zeros_like(c_not_idxs)])
            scores.append(eval_clf(clf_loo, X_test, y_test))

    return f_idx, C, scores


def analyze_features(features, word2idx, embeddings, clfs=None):
    """
    Compute metrics for all features.

    Arguments:
        features: dict of feature_name -> `Feature`
        word2idx: concept name -> concept id dict
        embeddings: numpy array of concept embeddings
        clfs: optional pretrained feature classifiers. If not provided, these
            will be trained on-the-spot.

    Returns:
        List of `AnalyzeResult` namedtuples for all features which satisfied
        internal constraints. Fields:
            feature: `Feature` tuple
            n_concepts: # concepts included in feature analysis
            clf: trained sklearn binary classifier for feature, accepting
                embeddings as input
            metric: summary goodness metric, where higher is better
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
    counts = Y.sum(axis=0)
    results = []
    for f_idx, f_name in tqdm(enumerate(usable_features),
                              total=len(usable_features),
                              desc="Training feature classifiers"):
        clf = None
        if clfs is not None:
            clf = clfs.get(f_name)
        if clf is None:
            clf = clf_base(C=Cs[f_name])
            clf.fit(X, Y[:, f_idx])

        preds = clf.predict(X)
        metric = metrics.f1_score(Y[:, f_idx], preds)

        results.append(AnalyzeResult(features[f_name], counts[f_idx],
                                     clf, metric))

    return results


def analyze_classifiers(analyze_results, all_embeddings, min_count=300):
    """
    Analyze the learned weights of feature classifiers.

    Arguments:
        analyze_results: list of AnalyzeResult tuples
    """

    clf_coefs = np.concatenate([result.clf.coef_ for result in analyze_results])
    clf_coefs /= np.linalg.norm(clf_coefs, axis=1, keepdims=True)
    all_embeddings.init_sims()
    sims = np.dot(clf_coefs, all_embeddings.syn0norm.T)

    # Cache counts for faster inner loop
    word_counts = {word.index: word.count
                   for word in all_embeddings.vocab.values()}
    word_counts = [word_counts[i] for i in range(len(word_counts))]

    lowercase = set(string.ascii_lowercase)

    nearby_words = {}
    for result, r_sims in tqdm(zip(analyze_results, sims),
                               desc="Analyzing classifiers",
                               total=len(analyze_results)):
        feature = result.feature.name
        clf = result.clf

        r_sims_sort = r_sims.argsort()[::-1]
        nearby_f = []
        for r_sim_idx in r_sims_sort:
            if len(nearby_f) == 50:
                break
            count = word_counts[r_sim_idx]
            if count is None or count < min_count:
                continue

            word = all_embeddings.index2word[r_sim_idx]
            if not word[0] in lowercase:
                continue
            nearby_f.append((word, r_sims[r_sim_idx]))

        nearby_words[feature] = nearby_f

    return nearby_words


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

    print("average domain pearson variance",
        sum(domain_p1_vars[d] for d in domain_p1_vars) / len(domain_p1_vars))

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
    ax.set_xlabel("ρ(" + PIVOT_FORMAL + "," + SOURCE_FORMAL + ")")
    ax.set_ylabel("ρ(" + PIVOT_FORMAL + ", WordNet)")
    ax.scatter(xs, ys, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], ys[i]), fontsize=20)

    plot_gaussian_contour(xs, ys, x_vars, y_vars)
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-%s.png" % (PEARSON1_NAME, PEARSON2_NAME))
    fig.savefig(fig_path, **SAVEFIG_KWARGS)

    # Plot feature metric vs. Pearson1

    fig = plt.figure()
    fig.suptitle("unified graph")
    ax = fig.add_subplot(111)
    ax.set_xlabel("ρ(" + PIVOT_FORMAL + "," + SOURCE_FORMAL + ")")
    ax.set_ylabel("feature fit")
    ax.scatter(xs, zs, c=cs, alpha=0.8)
    for i, d in enumerate(labels):
        ax.annotate(d, (xs[i], zs[i]))

    plot_gaussian_contour(xs, zs, x_vars, z_vars)
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified_domain-%s-feature.png" % PEARSON1_NAME)
    fig.savefig(fig_path, **SAVEFIG_KWARGS)

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
    fig.savefig(fig_path, **SAVEFIG_KWARGS)


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
    sns_plot.set(xlabel='Domain ID', ylabel='Median feature fit score')
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-domain.png" % PIVOT)
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(fig_path, **SAVEFIG_KWARGS)


def produce_unified_graph(vocab, features, feature_data, domain_concepts=None):
    concept_pearson1 = get_map_from_tsv(PEARSON1, "Concept", "correlation")
    concept_pearson2 = get_map_from_tsv(PEARSON2, 'Concept', 'correlation')
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

        print("%s\t%f" % (concept, np.median(weights)))

    zs = np.array(zs)
    zs *= 100

    concept_domains = {c: [d] for d, cs in domain_concepts.items() for c in cs}
    analyze_domains(labels, zs, concept_domains=concept_domains)

    # Render Z axis using colors
    colormap = plt.get_cmap("cool")
    zs_shift = (zs - zs.min()) / (zs.max() - zs.min())
    cs = colormap(zs_shift)

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
    ax.set_xlabel("m(" + PIVOT_FORMAL + "," + SOURCE_FORMAL + ")")
    ax.set_ylabel("m(" + PIVOT_FORMAL + ", WordNet)")
    ax.scatter(xs, ys, c=cs, alpha=0.8)
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
    fig.savefig(fig_path, **SAVEFIG_KWARGS)
    plt.close()

    # Plot feature metric vs. Pearson1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("m(" + PIVOT_FORMAL + "," + SOURCE_FORMAL + ")")
    ax.set_ylabel("Median feature fit score")
    ax.scatter(xs, zs, c=cs, alpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.png" % PEARSON1_NAME)
    fig.savefig(fig_path, **SAVEFIG_KWARGS)
    plt.close()

    # Plot feature metric vs. Pearson2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(PEARSON2_NAME)
    ax.set_ylabel("Median feature fit score")
    ax.scatter(ys, zs, c=cs, alpha=0.8)

    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "unified-%s-feature.png" % PEARSON2_NAME)
    fig.savefig(fig_path, **SAVEFIG_KWARGS)
    plt.close()


def cluster_metric_fn(x, y):
    x_weight, y_weight = x[0], y[0]
    x_emb, y_emb = x[1:], y[1:]

    emb_dist = distance.cosine(x_emb, y_emb)
    if np.isnan(x_weight) or np.isnan(y_weight):
        return emb_dist
    else:
        weight_dist = (x_weight - y_weight) ** 2
        return emb_dist + 50 * weight_dist


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
    concept_vals = {c: np.median(vals) for c, vals in concept_vals.items()}

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
        results.append((np.median(weights), np.var(weights), sib_cluster))

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
    fig.savefig(fig_path, **SAVEFIG_KWARGS)


def do_bootstrap_test(feature_groups, pop1, pop2, n_bootstrap_samples=10000,
                      percentiles=(5, 95)):
    """
    Do a percentile bootstrap test on the difference of medians among features
    from different groups of categories.

    The hypothesis here is:

        median(features from pop2) - median(features from pop1) > 0
    """

    # Concatenate all features from pop1, pop2 into flat groups.
    pop1_features = list(itertools.chain.from_iterable(
            feature_groups[group] for group in pop1))
    pop2_features = list(itertools.chain.from_iterable(
            feature_groups[group] for group in pop2))

    # Convert these into easy-to-use NP arrays..
    pop1_features = np.array([result.metric for result in pop1_features])
    pop2_features = np.array([result.metric for result in pop2_features])

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


def swarm_feature_cats(feature_groups, fcat_median):
    fcats_sorted = sorted(feature_groups.keys(), key=lambda k: fcat_median[k])
    x, y = [], []
    for fg in fcats_sorted:
        for result in feature_groups[fg]:
            if fg == "visual-form_and_surface":
                x.append("visual-\nform_and_surface")
            elif fg == "taste" or fg == 'smell' or fg == 'sound':
                x.append("taste/smell/\nsound")
            else:
                x.append(fg)

            y.append(result.metric)

    y = np.array(y)
    y *= 100
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns_plot = sns.swarmplot(x, y, ax=ax)
    sns_plot = sns.boxplot(x, y, showcaps=False,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':0}, ax=ax)
    sns_plot.set(xlabel='Feature category', ylabel='Feature fit score')
    fig_path = os.path.join(GRAPH_DIR, "feature-%s-%s-category.png" % (SOURCE, PIVOT))
    plt.tight_layout()
    fig = sns_plot.get_figure()
    fig.savefig(fig_path, **SAVEFIG_KWARGS)


def main():
    features, concepts = load_features_concepts()
    all_embeddings = load_all_embeddings()
    vocab, embeddings = load_filtered_embeddings(concepts, all_embeddings)
    word2idx = {w: i for i, w in enumerate(vocab)}

    clfs = None
    clf_path = Path(CLASSIFIER_OUTPUT)
    if clf_path.exists():
        with clf_path.open("rb") as clf_f:
            print("Loading classifiers from pickled dump.")
            clfs = pickle.load(clf_f)

    feature_data = analyze_features(features, word2idx, embeddings, clfs=clfs)
    feature_data = sorted(feature_data, key=lambda f: f.metric)

    fcat_mean = {}
    fcat_median = {}

    grouping_fns = {
        "br_label": lambda name: features[name].br_label,
        "first_word": lambda name: name.split("_")[0],
    }
    groups = {k: defaultdict(list) for k in grouping_fns}

    # Pickle classifiers
    if not clf_path.exists():
        with clf_path.open("wb") as clf_out:
            clfs = {result.feature.name: result.clf for result in feature_data}
            pickle.dump(clfs, clf_out)

    classifier_nearby = analyze_classifiers(feature_data, all_embeddings,
                                            min_count=MIN_WORD_COUNT)
    with open(CLASSIFIER_NEIGHBOR_OUTPUT, "w") as f:
        for result in feature_data:
            feature = result.feature.name
            nearby = classifier_nearby[feature]

            f.write("%s\t%.5f\n" % (feature, result.metric))
            for w, sim in nearby:
                f.write("\t%.5f\t%s\n" % (sim, w))

    # Output raw feature data and group features
    with open(FF_OUTPUT, "w") as ff_out:
        for result in feature_data:
            ff_out.write("%s\t%s\t%i\t%f\n" %
                         (result.feature.name, result.feature.br_label,
                          result.n_concepts, result.metric))

            for grouping_fn_name, grouping_fn in grouping_fns.items():
                grouping_fn = grouping_fns[grouping_fn_name]
                group = grouping_fn(result.feature.name)
                groups[grouping_fn_name][group].append(result)

    # Output really raw feature data
    with open(FF_ALL_OUTPUT, "w") as ff_out:
        for result in feature_data:
            concept_probs = result.clf.predict_proba(embeddings)[:, 1]
            assert len(concept_probs) == len(vocab)

            cf_sorted = sorted(zip(concept_probs, vocab), reverse=True)
            for concept_prob, concept in cf_sorted:
                is_positive = concept in result.feature.concepts
                ff_out.write("%s\t%s\t%f\t%i\n"
                             % (result.feature.name, concept,
                                concept_prob, is_positive))

    # Output grouped feature information
    with open(GROUP_OUTPUT, "w") as group_out:
        for grouping_fn_name, groups_result in sorted(groups.items()):
            group_out.write("\n\nGrouping by %s:\n" % grouping_fn_name)
            summary = {}
            for group_name, group_data in groups_result.items():
                scores = np.array([group_data_i.metric
                                   for group_data_i in group_data])
                n_entries = np.array([group_data_i.n_concepts
                                      for group_data_i in group_data])

                summary[group_name] = (len(group_data), np.mean(scores),
                                       np.percentile(scores, (0, 50, 100)),
                                       np.mean(n_entries))
            # Sort by median.
            summary = sorted(summary.items(), key=lambda x: x[1][2][1])

            group_out.write("%25s\tmu\tn\tmed\t\tmin\tmean\tmax\n" % "group")
            group_out.write(("=" * 100) + "\n")
            for label_group, (n, mean, pcts, n_concepts) in summary:
                group_out.write("%25s\t%.2f\t%3i\t%.5f\t\t%.5f\t%.5f\t%.5f\n"
                                % (label_group, n_concepts, n, pcts[1], pcts[0],
                                   mean, pcts[2]))
                fcat_mean[label_group] = mean
                fcat_median[label_group] = pcts[1]

    # Output per-concept scores
    with open(CONCEPT_OUTPUT, "w") as concept_f:
        for concept in vocab:
            metrics = [result.metric for result in feature_data
                       if concept in result.feature.concepts]
            if not metrics: continue

            c_score = np.median(metrics)
            concept_f.write("%s\t%f\n" % (concept, c_score))

    #produce_feature_fit_bars(groups["br_label"])
    if SOURCE == "cslb":
        do_bootstrap_test(groups["br_label"],
                          ["visual perceptual", "other perceptual"],
                          ["taxonomic", "functional"])
    elif SOURCE == "mcrae":
        do_bootstrap_test(groups["br_label"],
                          ["visual-motion", "visual-form_and_surface", "visual-colour",
                           "sound", "tactile", "smell", "taste"],
                          ["function", "taxonomic"])

    with plt.style.context({"font.size": 20, "axes.labelsize": 20, "xtick.labelsize": 17, "ytick.labelsize": 20}):
        swarm_feature_cats(groups["br_label"], fcat_median)

        feature_data = [(result.feature.name, result.n_concepts, result.metric)
                        for result in feature_data]
        domain_concepts = do_cluster(vocab, features, feature_data)

        produce_unified_graph(vocab, features, feature_data, domain_concepts=domain_concepts)
        produce_unified_domain_graph(vocab, features, feature_data, domain_concepts=domain_concepts)


if __name__ == "__main__":
    main()
