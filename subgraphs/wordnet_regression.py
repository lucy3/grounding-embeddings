"""
Regression model which predicts difference in WordNet metrics

Loads the WordNet matching metrics as computed by `wordnet_match` for multiple
data sources, and tries to predict the difference between the matching scores.
"""

from collections import defaultdict
import csv
import os.path

from nltk.corpus import wordnet as wn
import numpy as np
from sklearn import linear_model


CONC_BRM = "../mcrae/CONCS_brm.txt"
CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"
DOMAINS = set(["a_bird", "a_fish", "a_fruit", "a_mammal", \
    "a_musical_instrument", "a_tool", "a_vegetable", "an_animal"])

INPUT_FILE1 = "./all/hier_clust/wordnet_match_wikigiga.txt"
# INPUT_FILE1 = "./all/hier_clust/wordnet_match_cc.txt"

INPUT_FILE2 = "./all/hier_clust/wordnet_match_mcrae.txt"

# which column to use (i.e. which specific metric to use) from each file?
# zero-indexed --> METRIC_FIELD = 1 == the second column == the first metric
# column
METRIC_FIELD = 14


def load_metrics():
    """
    Load WordNet matching metrics for the two input files.
    """
    concept_metrics1 = {}
    metric_name = None
    with open(INPUT_FILE1, "r") as inp1:
        for i, line in enumerate(inp1):
            fields = line.strip().split("\t")
            if i == 0:
                metric_name = fields[METRIC_FIELD]
                print("Using column %i == %s" % (METRIC_FIELD, metric_name))
                continue

            concept, value = fields[0], fields[METRIC_FIELD]
            try:
                value = float(value)
            except:
                # n/a
                value = 0.0
            concept_metrics1[concept] = value

    concept_metrics2 = {}
    with open(INPUT_FILE2, "r") as inp2:
        for i, line in enumerate(inp2):
            fields = line.strip().split("\t")
            if i == 0:
                col_name = fields[METRIC_FIELD]
                if col_name != metric_name:
                    raise ValueError("mismatching column %i in file 2: %s != %s"
                                     % (METRIC_FIELD, col_name, metric_name))
                continue

            concept, value = fields[0], fields[METRIC_FIELD]
            try:
                value = float(value)
            except:
                # n/a
                value = 0.0
            concept_metrics2[concept] = value

    assert set(concept_metrics1.keys()) == set(concept_metrics2.keys())
    concept_metrics = {c: (m1, concept_metrics2[c]) for c, m1
                       in concept_metrics1.items()}

    return concept_metrics


def do_regression(reg_data, concept_stats):
    """
    Regress from concept data stored in `concept_stats` to targets.
    """
    N = len(reg_data)
    X, y = [], []
    for concept, target in reg_data:
        X.append([float(x) for x in concept_stats[concept]])
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    r2 = reg.score(X, y)
    params = reg.coef_
    return r2, params


def get_concept_stats():
    """
    Load auxiliary concept information from McRae data for use as regression
    features.
    """
    concept_stats = defaultdict(list)
    domains = {}
    prod_freqs = defaultdict(int)

    with open(CONCSTATS, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            prod_freqs[row["Concept"]] += int(row["Prod_Freq"])
            if row["Feature"] in DOMAINS:
                domains[row["Concept"]] = row["Feature"]

    with open(CONC_BRM, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            concept = row["Concept"]
            row_stats = (row["BNC"], row["Num_Feats_Tax"], row["Familiarity"],
                         prod_freqs[concept], len(wn.synsets(concept)))
            concept_stats[concept] = row_stats

    return concept_stats, domains


def augment_concept_stats(concept_stats, concept_domains):
    """
    Augment concept_stats dictionary with domain information.
    """
    # Build a canonicalized format for the domain space.
    all_domains = list(sorted(set(concept_domains.values())))

    ret = {}
    for concept in concept_stats:
        concept_domain = concept_domains.get(concept, None)
        domains = [1 if domain == concept_domain else 0
                   for domain in all_domains]
        ret[concept] = concept_stats[concept] + tuple(domains)

    return ret, all_domains


def main():
    concept_metrics = load_metrics()
    concept_stats, domains = get_concept_stats()

    # Generate regression targets.
    reg_data = [(concept, m1 - m2) for concept, (m1, m2)
                in concept_metrics.items()]

    print("File 1: ", os.path.basename(INPUT_FILE1))
    print("File 2: ", os.path.basename(INPUT_FILE2))

    print("Positive weight for a domain == matches are better in this domain \n"
          "for file 1 vs file 2.")

    # Baseline
    r2, _ = do_regression(reg_data, concept_stats)
    print("baseline regression: %5f" % r2)

    augmented_concept_stats, augmented_labels = \
            augment_concept_stats(concept_stats, domains)
    r2, weights = do_regression(reg_data, augmented_concept_stats)
    print("augmented regression: %5f" % r2)

    augmented_weights = weights[-len(augmented_labels):]
    augmented_weights = sorted(zip(augmented_weights, augmented_labels))
    from pprint import pprint
    pprint(list(augmented_weights))


if __name__ == '__main__':
    main()
