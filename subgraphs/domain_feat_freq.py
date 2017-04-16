
"""
Domain, average pearson value,
average wordnet value, and the percentage
of feature categories in it.
"""

CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
PEARSON = './all/pearson_corr/corr_mcrae_wikigiga.txt'
WORDNET = './all/hier_clust/wordnet_match_wikigiga.txt'
OUTPUT = './all/domain_feat_freq.txt'
GRAPH_DIR = './all/domain_feat_graphs'

import csv
import os.path

import get_domains
import numpy as np

def get_feat_freqs(concept_domains, domain_concepts):
    '''
    @inputs
    - concept_domains = {concept : [domains]}
    - domain_concepts = {domain: [concepts]}
    @outputs
    - domains = sorted list of domains
    - fcat_list = sorted list of feature categories
    - domain_matrix = numpy array of average production frequencies, where
    rows are domains and columns are fcat_list
    '''
    vocab_file = open(VOCAB, 'r')
    vocabulary = set()
    for line in vocab_file:
        vocabulary.add(line.strip())

    domains = sorted([d for d in domain_concepts.keys() if len(domain_concepts[d]) > 2])
    feature_cats = set()
    domain_feats = {d: [] for d in domains}
    with open(CONCSTATS, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row["Concept"] in vocabulary:
                if row["Concept"] == 'bluejay':     # weird edge case
                    c_domains = concept_domains['jaybird']
                else:
                    c_domains = concept_domains[row["Concept"]]
                for d in c_domains:
                    if d in domains:
                        domain_feats[d].append((row["BR_Label"], row["Prod_Freq"]))
                        feature_cats.add(row["BR_Label"])
    fcat_list = sorted(list(feature_cats))

    domain_matrix = np.zeros((len(domains), len(fcat_list))) # rows: domains, columns: feature categories
    for i in range(len(domains)):
        feats = domain_feats[domains[i]] # list of tuples (feature category, production frequency)
        for f in feats:
            domain_matrix[i][fcat_list.index(f[0])] += int(f[1])

    num_concepts = np.array([len(domain_concepts[domains[i]]) for i in range(len(domains))])
    domain_matrix = domain_matrix/num_concepts[:,None]

    return(domain_matrix, domains, fcat_list)

def get_average(input_file, c_string, value, concept_domains, domain_concepts):
    domain_average = {d: 0 for d in domain_concepts.keys()}
    with open(input_file, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row[c_string] == 'bluejay':      # weird edge case
                c_domains = concept_domains['jaybird']
            else:
                c_domains = concept_domains[row[c_string]]
            for d in c_domains:
                if row[value] == 'n/a':
                    row[value] = 0
                domain_average[d] += float(row[value])
    for d in domain_average:
        domain_average[d] /= len(domain_concepts[d])
    return domain_average

def render_graphs(domain_pearson, domain_wordnet, domains, domain_matrix, fcat_list,
                  colormap="cool"):
    import matplotlib.pyplot as plt

    xs = [domain_pearson[domain] for domain in domains]
    ys = [domain_wordnet[domain] for domain in domains]

    # Normalize each column to a range [0, 1].
    domain_matrix = (domain_matrix - domain_matrix.min(axis=0)) / (domain_matrix.max(axis=0) - domain_matrix.min(axis=0))

    colormap = plt.get_cmap(colormap)

    for j, fcat in enumerate(fcat_list):
        print(fcat)
        fig = plt.figure()
        fig.suptitle(fcat)

        ax = fig.add_subplot(111)
        ax.scatter(xs, ys)

        for i, (domain, x, y) in enumerate(zip(domains, xs, ys)):
            strength = domain_matrix[i, j]
            print("\t", domain, strength)
            ax.annotate(domain, (x, y), color=colormap(strength),
                        horizontalalignment="center",
                        verticalalignment="center")

        fig_path = os.path.join(GRAPH_DIR, fcat + ".png")
        fig.savefig(fig_path)

        print("\n\n")

def main():
    concept_domains = get_domains.get_concept_domains()
    domain_concepts = get_domains.get_domain_concepts()
    domain_matrix, domains, fcat_list = get_feat_freqs(concept_domains, domain_concepts)
    domain_pearson = get_average(PEARSON, 'Concept',
        'correlation', concept_domains, domain_concepts)
    domain_wordnet = get_average(WORDNET, 'concept',
        'dendrogram: 0.8; wordnet: 7', concept_domains, domain_concepts)

    render_graphs(domain_pearson, domain_wordnet, domains, domain_matrix, fcat_list)

    with open(OUTPUT, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['domain', 'num_concepts', 'pearson_avg', 'wordnet_avg'] + fcat_list)
        for i in range(len(domains)):
            writer.writerow([domains[i], len(domain_concepts[domains[i]]),
                domain_pearson[domains[i]], domain_wordnet[domains[i]]] +
                domain_matrix[i].tolist())

if __name__ == '__main__':
    main()
