from collections import defaultdict
import csv
from pathlib import Path

from nltk.corpus import wordnet as wn
import pattern.en as pattern
from pattern.text import TENSES


class cached_property(object):

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


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


def load_ppmi(ppmi_file):
    feature_ppmis = defaultdict(lambda: ([], []))
    concept_ppmis = defaultdict(lambda: ([], []))

    with open(ppmi_file, "r") as ppmi_f:
        for line in ppmi_f:
            fields = line.strip().split("\t")
            if len(fields) < 5: continue

            feature, _, concept, ppmi, is_positive = fields[:5]
            idx = 1 if is_positive == "True" else 0

            # Normalize feature name to match feature_fit output
            feature = feature.replace(" ", "_")
            feature_ppmis[feature][idx].append(float(ppmi))

            concept_ppmis[concept][idx].append(float(ppmi))

    return feature_ppmis, concept_ppmis


def load_feature_fit(fit_dir):
    feature_fits = {}
    feature_categories = {}

    with Path(fit_dir, "features.txt").open() as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 4: continue

            feature, category, score = fields[0], fields[1], fields[3]
            feature_fits[feature] = float(score)
            feature_categories[feature] = category

    return feature_fits, feature_categories


def load_concept_fit(fit_dir):
    concept_fits = {}

    with Path(fit_dir, "concepts.txt").open() as f:
        for line in f:
            concept, score = line.strip().split()
            concept_fits[concept] = float(score)

    return concept_fits


def get_map_from_tsv(tsv_path, k_col, v_col, cast=float):
    """
    Build a key-value map from two columns of a TSV file.

    Arguments:
        csv_path:
        k_col: Column which should yield keys of map. Values in this should
            column should be unique, otherwise things break!
        v_col: Column which should yield values of map.

    Returns:
        dict
    """

    ret = {}
    with open(tsv_path, "rU") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row[v_col] == "n/a":
                row[v_col] = 0
            ret[row[k_col]] = cast(row[v_col])

    return ret


def load_concept_corr(corr_path):
    return get_map_from_tsv(corr_path, "Concept", "correlation")
