import codecs
from collections import defaultdict, namedtuple
from pathlib import Path
from pprint import pprint
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import domain_feat_freq

# EMBEDDING_NAME = "mcrae" # McRae
# EMBEDDING_NAME = "glove.6B.300d" # Wikipedia 2014 + Gigaword 5
# EMBEDDING_NAME = "glove.840B.300d" # Common Crawl
INPUT = "../glove/%s.txt" % EMBEDDING_NAME
# INPUT = "./all/mcrae_vectors.txt"

FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
EMBEDDINGS = "./all/embeddings.%s.npy" % EMBEDDING_NAME

OUTPUT = "./all/feature_fit/mcrae_mcrae.txt"
PEARSON = './all/pearson_corr/corr_mcrae_wikigiga.txt'
WORDNET = './all/hier_clust/wordnet_match_mcrae.txt'
GRAPH_DIR = './all/feature_fit/mcrae'

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
                # Header row.
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


def analyze_feature(feature, features, word2idx, embeddings):
    """
    Compute metric for a given feature.

    Returns:
        feature_name:
        n_concepts: number of concepts which have this feature
        metric: goodness metric, where higher is better
    """
    # Fetch available embeddings.
    embeddings = [embeddings[word2idx[concept]]
                  for concept in features[feature].concepts
                  if concept in word2idx]
    if len(embeddings) < 3 or len(embeddings) > 7:
        return

    pca = PCA(n_components=1)
    pca.fit(embeddings)

    return feature, len(embeddings), pca.explained_variance_ratio_[0]


def plot_groups(label_groups):
    from matplotlib import pyplot as plt
    bins = np.linspace(0, 1, 30)
    for label_group, values in label_groups.items():
        if label_group in ["visual-colour", "function"]:
            plt.hist(values, bins=bins, alpha=0.5, label=label_group, normed=True)
    plt.legend(loc='upper left')
    plt.show()

def produce_domain_graphs(fcat_med):
    domain_pearson = domain_feat_freq.get_average(PEARSON, 'Concept',
        'correlation')
    domain_wordnet = domain_feat_freq.get_average(WORDNET, 'concept',
        'dendrogram: 0.8; wordnet: 6')
    domain_matrix, domains, fcat_list = domain_feat_freq.get_feat_freqs(weights=fcat_med)
    domain_feat_freq.render_graphs(GRAPH_DIR, domain_pearson, domain_wordnet,
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
    concept_pearson = get_values(PEARSON, 'Concept', 'correlation')
    concept_wordnet = get_values(WORDNET, 'concept', 'dendrogram: 0.8; wordnet: 6')
    assert concept_pearson.keys() == concept_wordnet.keys()
    vocab = sorted(concept_pearson.keys())
    concept_matrix, fcat_list = get_fcat_conc_freqs(vocab, weights=fcat_med)
    print(concept_matrix)

    xs = [concept_pearson[concept] for concept in vocab]
    ys = [concept_wordnet[concept] for concept in vocab]
    concept_matrix = (concept_matrix - concept_matrix.min(axis=0)) / (concept_matrix.max(axis=0)
        - concept_matrix.min(axis=0))
    colormap = plt.get_cmap("cool")

    for j, fcat in enumerate(fcat_list):
        print(fcat)
        fig = plt.figure()
        fig.suptitle(fcat+"-08-60-concepts-perc")

        ax = fig.add_subplot(111)
        c = colormap([concept_matrix[i, j] for i, concept in enumerate(vocab)])
        ax.scatter(xs, ys, [], c)

        fig_path = os.path.join(GRAPH_DIR, fcat)
        fig.savefig(fig_path + '-08-60-concepts-perc')

def main():
    features, concepts = load_features_concepts()
    vocab, embeddings = load_embeddings(concepts)
    word2idx = {w: i for i, w in enumerate(vocab)}

    feature_data = [analyze_feature(feature, features, word2idx, embeddings)
                    for feature in features]
    feature_data = sorted(filter(None, feature_data), key=lambda f: f[2])

    fcat_med = {} # {fcat: med}
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
                group = grouping_fn(name)
                groups[grouping_fn_name][group].append((score, n_entries))

        for grouping_fn_name, groups_result in groups.items():
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

    #produce_domain_graphs(fcat_med) # this calls functions in domain_feat_freq.py
    produce_concept_graphs(fcat_med) # this calls functions in here


if __name__ == "__main__":
    main()
