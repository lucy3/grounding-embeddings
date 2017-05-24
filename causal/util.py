from collections import defaultdict
from pathlib import Path


class cached_property(object):

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def load_ppmi(ppmi_file):
    feature_ppmis = defaultdict(lambda: ([], []))
    concept_ppmis = defaultdict(lambda: ([], []))

    with open(ppmi_file, "r") as ppmi_f:
        for line in ppmi_f:
            fields = line.strip().split("\t")
            if len(fields) < 4: continue

            feature, concept, ppmi, is_positive = fields[:4]
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

