from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import csv
from itertools import chain
from pathlib import Path
from pprint import pprint
import re

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


class Feature(object):
    def __init__(self, name, category):
        self.name = name
        self.category = category
        self.concepts = []
        self.alternatives = set()
        self.count = 0

    def process_description(self, desc):
        return " ".join([word for word in desc.split()
                         if word not in FEATURE_STOPWORDS])

    @property
    def processed_alternatives(self):
        return [self.process_description(alt)
                for alt in self.alternatives]

    @property
    def cooccur_targets(self):
        main_str = self.process_description(self.name)
        alt_strs = self.processed_alternatives

        ret = main_str.split()
        for alt_str in alt_strs:
            ret.extend(alt_str.split())
        return set(ret)


p = ArgumentParser()
p.add_argument("--norms-file", default="./cslb/norms.dat")
p.add_argument("--cooccur-file", default="./cooccur.npz")
p.add_argument("--vocab-file", default="./vocab.txt")
p.add_argument("--filtered-vocab-file", default="./vocab.keep.txt")
p.add_argument("--cooccur-pmi-file", default="./cooccur.pmi.npz")

p.add_argument("--mode", choices=["write-vocab", "ppmi"])

args = p.parse_args()


FEATURE_STOPWORDS = frozenset([
    "beh-", "inbeh-", "used", "to", "long", "eaten", "by", "found", "the",
    "made", "is", "has", "a", "in", "worn", "on", "of", "an", "for", "shaped",
    "like", "does", "not", "similar", "get", "attached", "along", "at", "with",
    "associated", "you", "contains", "as", "go", "and", "what", "your", "make",
    "look", "function", "up", "enable", "keep", "over", "leave", "allow", "when",
    "use", "aid", "capable", "lot", "considered", "its", "things", "come",
    "comes", "if", "more", "than", "onto", "into", "be", "very", "goes",
    "about", "makes", "from", "or", "able", "something", "takes", "time",
    "getting", "most", "being", "become", "variety", "many", "around", "kind",
    "can", "which", "it", "we", "take", "back", "consists", "consist",
    "several", "much", "can", "cause", "popular", "type", "feel", "outside",
    "needed", "need", "versatile", "containing", "some", "any", "all",
    "different", "looks", "connotation", "vary", "one", "single", "comprise",
    "comprised", "symbol", "one", "gets", "too", "let", "inside", "have",
    "material", "act", "acts", "said", "regarded", "source", "create", "so",
    "ability", "while", "that", "uses", "available", "characterised",
    "contain", "anymore", "been", "may", "no", "longer", "now", "was",
    "form", "under", "eat", "feed", "likes", "see", "x", "useful", "given",
    "provided", "can't", "covers", "coloured", "pieces", "cases", "lives",
    "originates", "live", "kept", "common", "added", "content", "their", "my",
    "but", "best", "good", "doing", "are", "three", "two",
])


def load_lil(filename):
    coo_data = np.load(filename)
    coo = coo_matrix((coo_data["data"], (coo_data["row"], coo_data["col"])),
                     shape=coo_data["shape"])
    return coo.tolil()

def save_lil(filename, lil):
    coo = lil.tocoo()
    np.savez(filename, data=coo.data, row=coo.row, col=coo.col, shape=coo.shape)


def load_cooccur():
    with open(args.vocab_file, "r") as vocab_f:
        vocab = [line.strip().split()[0] for line in vocab_f]

    pmi_path = Path(args.cooccur_pmi_file)
    if pmi_path.exists():
        cooccur = load_lil(args.cooccur_pmi_file)
    else:
        cooccur = load_lil(args.cooccur_file)
        cooccur = convert_pmi(cooccur)
        save_lil(args.cooccur_pmi_file, cooccur)
    return vocab, cooccur


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


def convert_pmi(cooccur):
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
            ret[i_x, i_y] = np.log(p_yx) - np.log(p_y)

    return ret


def do_ppmi_analysis(vocab, features, ppmi):
    # NB: not PPMI right now, but PMI
    vocab2idx = {word: i for i, word in enumerate(vocab)}

    for feature_name in sorted(features.keys()):
        feature = features[feature_name]

        for concept in sorted(feature.concepts):
            try:
                c_idx = vocab2idx[concept]
            except KeyError: continue

            ppmis = []
            for alt_tok in feature.cooccur_targets:
                try:
                    alt_idx = vocab2idx[alt_tok]
                except KeyError: continue

                # Only count if it's actually in the sparse matrix
                if alt_idx in ppmi.rows[c_idx]:
                    ppmis.append(ppmi[c_idx, alt_idx])

            print("%s\t%s\t%f\t%s" % (feature_name, concept, np.mean(ppmis), " ".join(feature.cooccur_targets)))


def main():
    features, concepts = load_features_concepts()

    if args.mode == "write-vocab":
        # Write a filtered vocab with the relevant words.
        words = set()
        for feature in features.values():
            words |= set(feature.cooccur_targets)

        with open(args.filtered_vocab_file, "w") as filtered_vocab_f:
            for word in sorted(words):
                filtered_vocab_f.write(word + "\n")
    elif args.mode == "ppmi":
        vocab, cooccur = load_cooccur()
        do_ppmi_analysis(vocab, features, cooccur)


if __name__ == '__main__':
    main()
