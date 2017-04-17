import codecs
from collections import defaultdict, namedtuple
from pathlib import Path
from pprint import pprint

import numpy as np

EMBEDDING_NAME = "mcrae" # McRae, even though it's not an embedding
# EMBEDDING_NAME = "glove.6B.300d" # Wikipedia 2014 + Gigaword 5
# EMBEDDING_NAME = "glove.840B.300d" # Common Crawl
# INPUT = "../glove/%s.txt" % EMBEDDING_NAME
INPUT = "./all/mcrae_vectors.txt"

FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
EMBEDDINGS = "./all/embeddings.%s.npy" % EMBEDDING_NAME
OUTPUT = "./all/feature_fit.txt"


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
    # Fetch available embeddings.
    embeddings = [embeddings[word2idx[concept]]
                  for concept in features[feature].concepts
                  if concept in word2idx]
    if len(embeddings) < 4 or len(embeddings) > 7:
        return
    embeddings = np.asarray(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    mean = np.mean(embeddings, axis=0)
    mean /= np.linalg.norm(mean)

    avg_dot = np.dot(embeddings, mean).mean()
    return (feature, len(embeddings), avg_dot)


def plot_groups(label_groups):
    from matplotlib import pyplot as plt
    bins = np.linspace(0, 1, 30)
    for label_group, values in label_groups.items():
        if label_group in ["visual-colour", "function"]:
            plt.hist(values, bins=bins, alpha=0.5, label=label_group, normed=True)
    plt.legend(loc='upper left')
    plt.show()


def main():
    features, concepts = load_features_concepts()
    vocab, embeddings = load_embeddings(concepts)
    word2idx = {w: i for i, w in enumerate(vocab)}

    feature_data = [analyze_feature(feature, features, word2idx, embeddings)
                    for feature in features]
    feature_data = sorted(filter(None, feature_data), key=lambda f: f[2])

    grouping_fns = {
        "br_label": lambda name: features[name].br_label,
        "first_word": lambda name: name.split("_")[0],
    }
    groups = {k: defaultdict(list) for k in grouping_fns}
    for name, n_entries, score in feature_data:
        print("%40s\t%25s\t%i\t%f" % (name, features[name].br_label,
                                      n_entries, score))

        for grouping_fn_name, grouping_fn in grouping_fns.items():
            group = grouping_fn(name)
            groups[grouping_fn_name][group].append((score, n_entries))

    # plot_groups(label_groups)

    for grouping_fn_name, groups_result in groups.items():
        print("\n\nGrouping by %s:" % grouping_fn_name)
        summary = {}
        for name, data in groups_result.items():
            data = np.array(data)
            scores = data[:, 0]
            n_entries = data[:, 1]
            summary[name] = (len(data), np.mean(scores), np.percentile(scores, (0, 50, 100)),
                             np.mean(n_entries))
        summary = sorted(summary.items(), key=lambda x: x[1][2][1])

        print("%25s\tmu\tn\tmed\t\tmin\tmean\tmax" % "group")
        print("=" * 100)
        for label_group, (n, mean, pcts, n_concepts) in summary:
            print("%25s\t%.2f\t%3i\t%.5f\t\t%.5f\t%.5f\t%.5f"
                  % (label_group, n_concepts, n, pcts[1], pcts[0], mean, pcts[2]))


if __name__ == "__main__":
    main()
