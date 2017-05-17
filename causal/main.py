from argparse import ArgumentParser
from collections import defaultdict, namedtuple
import csv
from itertools import chain
from pathlib import Path
from pprint import pprint
import re

from nltk.corpus import wordnet as wn
import numpy as np
import pattern.en as pattern
from pattern.text import TENSES
from scipy.sparse import coo_matrix, lil_matrix


from util import cached_property


class Feature(object):
    def __init__(self, name, category):
        self.name = name
        self.category = category
        self.concepts = []
        self.alternatives = set()
        self.count = 0

    def process_description(self, desc):
        return " ".join([word for word in desc.replace("_", " ").split()
                         if word not in FEATURE_STOPWORDS])

    @property
    def processed_alternatives(self):
        return [self.process_description(alt)
                for alt in self.alternatives]

    @cached_property
    def cooccur_targets(self):
        main_str = self.process_description(self.name)
        alt_strs = self.processed_alternatives

        ret = main_str.split()
        for alt_str in alt_strs:
            words = alt_str.split()
            ret.extend(words)

            # Augment with related words, drawn from WordNet
            for word in words:
                related = [related_word for related_word, p in morphify(word)
                           if p > 0.5]
                ret.extend(related)

        new_ret = set()
        # Add all the inflections!!!
        for word in ret:
            new_ret.add(word)

            # Plural+singular
            new_ret.add(pattern.pluralize(word))
            new_ret.add(pattern.singularize(word))

            # comparatives
            comparative = pattern.comparative(word)
            if "more" not in comparative:
                new_ret.add(comparative)
            superlative = pattern.superlative(word)
            if "most" not in superlative:
                new_ret.add(superlative)

            for id, tense in TENSES.items():
                if id is None: continue
                new_ret.add(pattern.conjugate(word, tense))

        return set(new_ret) - set([None])


p = ArgumentParser()
p.add_argument("--norms-file", default="./cslb/norms.dat")
p.add_argument("--cooccur-file", default="./cooccur.npz")
p.add_argument("--vocab-file", default="./vocab.txt")
p.add_argument("--filtered-vocab-file", default="./vocab.keep.txt")
p.add_argument("--cooccur-pmi-file", default="./cooccur.pmi.npz")

p.add_argument("--mode", choices=["write-vocab", "ppmi"])

args = p.parse_args()


# TODO maybe stop doing this brute-force and just weight PMIs by IDF within
# the CSLB corpus?
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
    "but", "best", "good", "doing", "are", "three", "two", "through", "other",
    "varying", "number", "needs", "always", "set", "there", "properly", "way",
    "do", "who", "those", "out", "off", "near", "high", "close", "above",
    "locations", "where", "provides", "object", "objects", "item", "items",
    "aimed", "designed", "seen", "attracted", "tend", "other", "help",
    "demonstrate", "demonstrates", "requires", "perform", "carries", "out",
    "meant", "provide", "one's", "least", "set", "such",
])


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
    pmi_path = Path(args.cooccur_pmi_file)
    if pmi_path.exists():
        cooccur = load_lil(args.cooccur_pmi_file)
    else:
        cooccur = load_lil(args.cooccur_file)
        cooccur = convert_pmi(cooccur)
        save_lil(args.cooccur_pmi_file, cooccur)
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


def morphify(word):
    # Stolen from http://stackoverflow.com/a/29342127/176075
    # slightly modified..
    synsets = wn.synsets(word)

    # Word not found
    if not synsets:
        return []

    # Get all  lemmas of the word
    lemmas = [l for s in synsets for l in s.lemmas()]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms())
                                    for l in lemmas]

    # filter only the targeted pos
    related_lemmas = [l for drf in derivationally_related_forms
                      for l in drf[1]]

    # Extract the words from the lemmas
    words = [l.name() for l in related_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result


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


def do_ppmi_analysis(vocab, features, ppmi):
    # NB: not PPMI right now, but PMI
    vocab2idx = {word: i for i, word in enumerate(vocab)}

    for feature_name in sorted(features.keys()):
        feature = features[feature_name]

        for concept in sorted(feature.concepts):
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

            print("%s\t%s\t%f\t%s" % (feature_name, concept, np.median(ppmis), " ".join(feature.cooccur_targets)))


def main():
    vocab = load_vocab()
    features, concepts = load_features_concepts()

    if args.mode == "write-vocab":
        write_vocab(vocab, features, concepts)
    elif args.mode == "ppmi":
        cooccur = load_cooccur()
        do_ppmi_analysis(vocab, features, cooccur)


if __name__ == '__main__':
    main()
