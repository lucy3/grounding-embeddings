import codecs
from collections import defaultdict
from pathlib import Path

import numpy as np


EMBEDDING_NAME = "glove.6B.300d"
GLOVE_INPUT = "../glove/%s.txt" % EMBEDDING_NAME

FEATURES = "./all/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
EMBEDDINGS = "./all/embeddings.%s.npy" % EMBEDDING_NAME
OUTPUT = "./all/feature_fit.txt"


def load_embeddings(c2f):
    if Path(EMBEDDINGS).is_file():
        embeddings = np.load(EMBEDDINGS)

        assert Path(VOCAB).is_file()
        with open(VOCAB, "r") as vocab_f:
            vocab = [line.strip() for line in vocab_f]
        assert len(embeddings) == len(vocab), "%i %i" % (len(embeddings), len(vocab))
    else:
        vocab, embeddings = [], []
        with codecs.open(GLOVE_INPUT, "r", encoding="utf-8") as glove_f:
            for line in glove_f:
                fields = line.strip().split()
                word = fields[0]
                if word in c2f:
                    vec = [float(x) for x in fields[1:]]
                    embeddings.append(vec)
                    vocab.append(word)

        embeddings = np.array(embeddings)
        np.save(EMBEDDINGS, embeddings)

        with open(VOCAB, "w") as vocab_f:
            vocab_f.write("\n".join(vocab))

    return vocab, embeddings


def load_features_concepts():
    """
    Returns:
        c2f: dict concept -> set(features)
        f2c: dict feature -> set(concepts)
    """
    c2f, f2c = defaultdict(set), defaultdict(set)

    with open(FEATURES, "r") as features_f:
        for line in features_f:
            concept, feature = line.strip().split("\t")[:2]

            c2f[concept].add(feature)
            f2c[feature].add(concept)

    lengths = [len(xs) for xs in f2c.values()]
    from collections import Counter
    from pprint import pprint
    print("# of features with particular number of associated concepts:")
    pprint(Counter(lengths))

    return c2f, f2c


def analyze_feature(feature, f2c, word2idx, embeddings):
    # Fetch available embeddings.
    embeddings = [embeddings[word2idx[concept]]
                  for concept in f2c[feature]
                  if concept in word2idx]
    if len(embeddings) <= 1:
        return
    embeddings = np.asarray(embeddings)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    mean = np.mean(embeddings, axis=0)
    mean /= np.linalg.norm(mean)

    avg_dot = np.dot(embeddings, mean).mean()
    return (feature, len(embeddings), avg_dot)


def main():
    c2f, f2c = load_features_concepts()
    vocab, embeddings = load_embeddings(c2f)
    word2idx = {w: i for i, w in enumerate(vocab)}

    feature_data = [analyze_feature(feature, f2c, word2idx, embeddings)
                    for feature in f2c]
    feature_data = sorted(filter(None, feature_data), key=lambda f: f[2])

    for name, n_entries, score in feature_data:
        print("%40s\t%i\t%f" % (name, n_entries, score))

if __name__ == "__main__":
    main()
