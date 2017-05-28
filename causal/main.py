from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import csv
from itertools import chain
from pathlib import Path
from pprint import pprint
import re
import sys

import numpy as np
import pattern.en as pattern
from scipy.sparse import coo_matrix, lil_matrix

from util import Feature


p = ArgumentParser()
p.add_argument("--norms-file", default="./cslb/norms.dat")
p.add_argument("--cooccur-file", default="./cooccur.npz")
p.add_argument("--vocab-file", default="./vocab.txt")
p.add_argument("--filtered-vocab-file", default="./vocab.keep.txt")
p.add_argument("--cooccur-ppmi-file", default="./cooccur.ppmi.npz")

p.add_argument("--mode", choices=["write-vocab", "ppmi"])

args = p.parse_args()


def load_lil(filename):
    coo_data = np.load(filename)
    coo = coo_matrix((coo_data["data"], (coo_data["row"], coo_data["col"])),
                     shape=coo_data["shape"])
    return coo.tolil()

def save_lil(filename, lil):
    coo = lil.tocoo()
    np.savez(filename, data=coo.data, row=coo.row, col=coo.col, shape=coo.shape)


def load_vocab():
    with open(args.vocab_file, "r") as vocab_f:
        vocab = [line.strip().split()[0] for line in vocab_f]
    return vocab


def load_cooccur():
    pmi_path = Path(args.cooccur_ppmi_file)
    if pmi_path.exists():
        cooccur = load_lil(args.cooccur_ppmi_file)
    else:
        cooccur = load_lil(args.cooccur_file)
        cooccur = convert_ppmi(cooccur)
        save_lil(args.cooccur_ppmi_file, cooccur)
    return cooccur


def load_features_concepts(min_concepts=5):
    features = {}
    concepts = defaultdict(list)

    with open(args.norms_file, "r") as norms_f:
        csv_f = csv.DictReader(norms_f, delimiter="\t")
        for row in csv_f:
            feature_name = row["feature"]
            if feature_name not in features:
                features[feature_name] = Feature(feature_name,
                                                 row["feature type"])

            alts = set([x.strip() for x in row["feature alternatives"].split(";")])
            features[feature_name].alternatives |= alts

            features[feature_name].concepts.append(row["concept"])
            features[feature_name].count += 1

    # Threshold features by # concepts
    features = {name: f for name, f in features.items()
                if f.count >= min_concepts}
    for feature in features.values():
        for concept in feature.concepts:
            concepts[concept].append(feature.name)

    return features, concepts


def convert_ppmi(cooccur):
    """
    Convert the co-occurrence matrix to PPMI
    """
    vocab_size = cooccur.shape[0]
    N = cooccur.sum()

    # calc unigram probabilities
    uni_probs = {idx: sum(data) / N
                 for idx, data in enumerate(cooccur.data)
                 if data}
    print(len(uni_probs))

    ret = lil_matrix(cooccur.shape)
    for i_x, p_x in uni_probs.items():
        # Calculate p(y|x) p(x) for all y
        data_x = cooccur.data[i_x]
        count_x_all = sum(data_x)
        for i_y, count_xy in zip(cooccur.rows[i_x], data_x):
            p_yx = count_xy / count_x_all
            p_y = uni_probs[i_y]
            ret[i_x, i_y] = max(0, np.log(p_yx) - np.log(p_y))

    return ret


def write_vocab(glove_vocab, features, concepts):
    # Write a filtered vocab with the relevant words.
    words = set()
    for feature in features.values():
        words |= set(feature.cooccur_targets)

    for concept in concepts.keys():
        words.add(concept)

    words = words & set(glove_vocab)

    with open(args.filtered_vocab_file, "w") as filtered_vocab_f:
        for word in sorted(words):
            filtered_vocab_f.write(word + "\n")


def do_ppmi_analysis(vocab, features, concepts, ppmi):
    # NB: not PPMI right now, but PMI
    vocab2idx = {word: i for i, word in enumerate(vocab)}

    for feature_name in sorted(features.keys()):
        feature = features[feature_name]
        concept_scores = {}

        for concept in concepts:
            try:
                c_idxs = [vocab2idx[concept]]
            except KeyError: continue

            try:
                c_idxs.append(vocab2idx[pattern.pluralize(concept)])
            except KeyError: pass

            ppmis = []
            for alt_tok in feature.cooccur_targets:
                try:
                    alt_idx = vocab2idx[alt_tok]
                except KeyError: continue

                for c_idx in c_idxs:
                    # Only count if it's actually in the sparse matrix
                    if alt_idx in ppmi.rows[c_idx]:
                        ppmis.append(ppmi[c_idx, alt_idx])

            is_positive = concept in feature.concepts
            if len(ppmis) > 0:
                score = np.max(ppmis)
            else:
                score = -np.inf
            concept_scores[concept] = (score, is_positive)

        concept_scores = sorted(concept_scores.items(), key=lambda x: x[1][0],
                                reverse=True)
        for concept, (score, is_positive) in concept_scores:
            print("%s\t%s\t%s\t%f\t%s" % (feature_name, feature.category,
                                          concept, score, is_positive))

        print()
        sys.stdout.flush()


def main():
    vocab = load_vocab()
    features, concepts = load_features_concepts()

    if args.mode == "write-vocab":
        write_vocab(vocab, features, concepts)
    elif args.mode == "ppmi":
        cooccur = load_cooccur()
        do_ppmi_analysis(vocab, features, concepts, cooccur)


if __name__ == '__main__':
    main()
