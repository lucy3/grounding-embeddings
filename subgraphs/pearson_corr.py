"""
Neighbor-distance correlation between two models

Also outputs the following stats:
BNC frequency (BNC_freq)
number of features (num_feats_tax)
familiarity (familiarity)
total # of features produced by participants (tot_num_feats)
# of WordNet senses (polysemy)

And at the bottom of the txt,
Average correlation in domains based on taxonomic features
with >15 associated concepts.
"""

import csv
from collections import defaultdict
import itertools
import operator

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from nltk.corpus import wordnet as wn

VOCAB = "./all/vocab.txt"
INPUT_FILE1 = "./all/sim_mcrae.txt"
INPUT_FILE2 = "./all/sim_glove_tw.txt"
OUTPUT_FILE = "./all/pearson_corr/corr_mcrae_tw.txt"
CONC_BRM = "../mcrae/CONCS_brm.txt"
CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"
DOMAINS = set(["a_bird", "a_fish", "a_fruit", "a_mammal", \
    "a_musical_instrument", "a_tool", "a_vegetable", "an_animal"])

def get_cosine_dist(input_file):
    """
    @output:
    - d: {(concept1, concept2) tuple : distance as a float}
    """
    d = defaultdict(float)
    word_sim = open(input_file, 'r')
    for line in word_sim:
        pair = tuple(line.split()[:2])
        dist = float(line.split()[2])
        d[pair] = dist
    return d

def get_neighbor_distance(input_file, vocabulary):
    """
    @input:
    - input_file: string name of an input file
    - vocabulary: set of concepts
    @output:
    - neighbor_distance: {concept: list of float distances to all other concepts}
    """
    cosine_dist = get_cosine_dist(input_file)
    neighbor_distance = {k: [0] * len(vocabulary) for k in vocabulary}
    for concept in vocabulary:
        for i in range(len(vocabulary)):
            neighbor = vocabulary[i]
            if (concept, neighbor) in cosine_dist:
                neighbor_distance[concept][i] = cosine_dist[(concept, neighbor)]
            elif (neighbor, concept) in cosine_dist:
                neighbor_distance[concept][i] = cosine_dist[(neighbor, concept)]
    return neighbor_distance

def get_mcrae_freq(pearson_co):
    """
    Prepares information to be written into
    this program's output file in a not-the-most elegant manner.

    @input:
    - pearson_co: {concept: pearson correlation float}
    @output:
    - concept_stats: {concept: tab-deliminated string of stats
    for later writing to a file}
    - average_in_domain: {domain string: average pearson correlation}
    - concept_features: {concept: set of feature strings}
    """
    concept_stats = defaultdict(list)
    concept_features = defaultdict(set)
    prod_freqs = defaultdict(int)
    sum_in_domain = defaultdict(float)
    count_in_domain = defaultdict(int)
    with open(CONCSTATS, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            prod_freqs[row["Concept"]] += int(row["Prod_Freq"])
            concept_features[row["Concept"]].add(row["Feature"])
            if row["Feature"] in DOMAINS:
                sum_in_domain[row["Feature"]] += pearson_co[row["Concept"]]
                count_in_domain[row["Feature"]] += 1

    with open(CONC_BRM, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            concept = row["Concept"]
            row_stats = (row["BNC"], row["Num_Feats_Tax"], row["Familiarity"],
                         prod_freqs[concept], len(wn.synsets(concept)))
            concept_stats[concept] = row_stats

    average_in_domain = defaultdict(float)
    for key in sum_in_domain:
        average_in_domain[key] = sum_in_domain[key]/count_in_domain[key]

    return concept_stats, average_in_domain, concept_features


def do_regression(sorted_pearson, concept_stats):
    """
    Regress from concept data stored in `concept_stats` to Pearson correlation
    values.
    """
    N = len(sorted_pearson)
    X, y = [], []
    for concept, corr in sorted_pearson:
        X.append([float(x) for x in concept_stats[concept]])
        y.append(corr)

    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    r2 = reg.score(X, y)
    return r2


def augment_concept_stats(concept_stats, concept_features):
    """
    Augment concept_stats dictionary with feature norm information.
    """
    # Build a canonicalized format for the feature norm space.
    all_features = list(sorted(set(itertools.chain.from_iterable(concept_features.values()))))
    feature2idx = {feature: idx for idx, feature in enumerate(all_features)}

    ret = {}
    for concept in concept_stats:
        associated_features = set(concept_features[concept])
        associated_features = [1 if feature in associated_features else 0
                               for feature in all_features]
        ret[concept] = concept_stats[concept] + tuple(associated_features)

    return ret


def main():
    # get vocabulary
    vocab_file = open(VOCAB, 'r')
    vocabulary = []
    for line in vocab_file:
        vocabulary.append(line.strip())

    # get pearson correlation b/t two datasets and other stats
    neighbor_dist1 = get_neighbor_distance(INPUT_FILE1, vocabulary)
    neighbor_dist2 = get_neighbor_distance(INPUT_FILE2, vocabulary)
    pearson_co = defaultdict(float)
    for concept in vocabulary:
            pearson_co[concept] = pearsonr(neighbor_dist1[concept], neighbor_dist2[concept])[0]
    sorted_pearson = sorted(pearson_co.items(), key=operator.itemgetter(1))
    concept_stats, average_in_domain, concept_features = \
            get_mcrae_freq(pearson_co)

    # Attempt a baseline regression.
    r2 = do_regression(sorted_pearson, concept_stats)
    print("baseline regression: %5f" % r2)

    augmented_concept_stats = augment_concept_stats(concept_stats, concept_features)
    r2 = do_regression(sorted_pearson, augmented_concept_stats)
    print("augmented regression: %5f" % r2)

    # write everything to an output file
    output = open(OUTPUT_FILE, 'w')
    output.write('Concept\tcorrelation\tBNC_freq\t' +
        'num_feats_tax\tfamiliarity\ttot_num_feats\tpolysemy\n')
    for pair in sorted_pearson:
        row_stats = "\t".join(str(stat) for stat in concept_stats[pair[0]])
        output.write(pair[0] + '\t' + str(pair[1]) + '\t' + row_stats + '\n')
    for tax_feature in sorted(average_in_domain.keys()):
        output.write("\n" + tax_feature + "\t" + str(average_in_domain[tax_feature]) + "\n")
    output.close()

if __name__ == '__main__':
    main()
