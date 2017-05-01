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
import operator

import numpy as np
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import get_domains
import math
from collections import Counter

VOCAB_SOURCE = "cslb"
SOURCE = "cslb"
PIVOT = "wikigiga"

VOCAB = "./all/vocab_%s.txt" % VOCAB_SOURCE
INPUT_FILE1 = "./all/sim_%s_%s.txt" % (VOCAB_SOURCE, SOURCE)
INPUT_FILE2 = "./all/sim_%s_%s.txt" % (VOCAB_SOURCE, PIVOT)
OUTPUT_FILE = "./all/pearson_corr/%s/corr_%s_%s.txt" % (VOCAB_SOURCE, SOURCE, PIVOT)

if VOCAB_SOURCE == "mcrae":
    CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"
elif VOCAB_SOURCE == "cslb":
    CONCSTATS = "../cslb/norms.dat"

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
    - domains: {concept: domain string}
    """
    concept_stats = defaultdict(list)
    prod_freqs = defaultdict(int)
    num_feats = defaultdict(int)
    if VOCAB_SOURCE == "mcrae":
        concept_header = 'Concept'
        pf_header = 'Prod_Freq'
    elif VOCAB_SOURCE == "cslb":
        concept_header = 'concept'
        pf_header = 'pf'
    with open(CONCSTATS, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                prod_freqs[row[concept_header]] += int(row[pf_header])
                num_feats[row[concept_header]] += 1

    assert len(prod_freqs.keys()) == len(num_feats.keys())

    wordcounts = Counter(brown.words())
    for concept in prod_freqs.keys():
        if concept == 'bluejay':
            senses = wn.synsets('jaybird')
        elif concept == 'rollerskate':
            senses = wn.synsets('roller_skate')
        elif concept == 'wetsuit':
            senses = wn.synsets('wet_suit')
        elif concept == 'yoyo':
            senses = wn.synsets('yo-yo')
        elif concept == 'deckchair':
            senses = wn.synsets('deck_chair')
        else:
            senses = wn.synsets(concept)
        count = wordcounts[concept]
        if count == 0:
            frequency = 0
        else:
            frequency = math.log(count)
        if len(senses) == 0:
            row_stats = (0, num_feats[concept],
                    prod_freqs[concept], 0)
        else:
            row_stats = (frequency,
                num_feats[concept], prod_freqs[concept], len(senses))
        concept_stats[concept] = row_stats

    concept_domains = get_domains.get_concept_domains()
    sum_in_domain = defaultdict(float)
    count_in_domain = defaultdict(int)
    domain_concepts = get_domains.get_domain_concepts()
    for domain in domain_concepts:
    	cons = domain_concepts[domain]
    	for c in cons:
    		sum_in_domain[domain] += pearson_co[c]
    	count_in_domain[domain] = len(cons)
    average_in_domain = defaultdict(float)
    for key in sum_in_domain:
        average_in_domain[key] = sum_in_domain[key]/count_in_domain[key]

    return concept_stats, average_in_domain, concept_domains


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
    params = reg.coef_
    return r2, params


def augment_concept_stats(concept_stats, concept_domains):
    """
    Augment concept_stats dictionary with domain information.
    """
    # Build a canonicalized format for the domain space.
    all_domains = list(sorted(set([item for sublist in concept_domains.values()
        for item in sublist])))

    ret = {}
    for concept in concept_stats:
        concept_domain = concept_domains.get(concept, None)
        domains = [0 if concept_domain is None or domain not in concept_domain
            else 1 for domain in all_domains]
        ret[concept] = concept_stats[concept] + tuple(domains)

    return ret, all_domains


def main():
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

    concept_stats, average_in_domain, domains = \
            get_mcrae_freq(pearson_co)

    # Attempt a baseline regression.
    r2, _ = do_regression(sorted_pearson, concept_stats)
    print("baseline regression: %5f" % r2)

    augmented_concept_stats, augmented_labels = \
            augment_concept_stats(concept_stats, domains)
    r2, weights = do_regression(sorted_pearson, augmented_concept_stats)
    print("augmented regression: %5f" % r2)

    augmented_weights = weights[-len(augmented_labels):]
    augmented_weights = sorted(zip(augmented_weights, augmented_labels))
    from pprint import pprint
    pprint(list(augmented_weights))

    # Print average correlations among domains
    for tax_feature in sorted(average_in_domain.keys()):
        print(str(tax_feature) + "\t" + str(average_in_domain[tax_feature]))

    # write everything to an output file
    output = open(OUTPUT_FILE, 'w')
    headers = ["Concept", "correlation", "log_brown_freq", "num_feats",
               "tot_prod_freq", "polysemy"]
    headers += map(str, augmented_labels)
    output.write("%s\n" % "\t".join(headers))
    for pair in sorted_pearson:
        row_stats = "\t".join(str(stat) for stat in augmented_concept_stats[pair[0]])
        output.write(pair[0] + '\t' + str(pair[1]) + '\t' + row_stats + '\n')
    output.close()

if __name__ == '__main__':
    main()
