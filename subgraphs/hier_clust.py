"""
Hierarchical clustering

Legend:
birds = red
fishes = olive
fruits = green
mammals = cyan
musical instruments = yellow
vegs = purple

Animal was not used as a labeling domain due to
tremendous overlap with bird, fish, and mammal.
Note: a tomato is both a fruit and a vegetable
but it could only take on one color label.
"""

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from pylab import savefig
import csv

VOCAB = "./all/vocab.txt"
DOMAINS = ["a_bird", "a_fish", "a_fruit", "a_mammal", \
	"a_musical_instrument", "a_tool", "a_vegetable"]
CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"

INPUT = "../glove/glove.6B.300d.txt" # Wikipedia 2014 + Gigaword 5
OUTPUT = "./all/hier_clust/dendrogram_wikigiga_domains.pdf"

# INPUT = "../glove/glove.840B.300d.txt" # Common Crawl
# OUTPUT = "./all/hier_clust/dendrogram_cc_domains.pdf"

# INPUT = "./all/mcrae_vectors.txt" # McRae
# OUTPUT = "./all/hier_clust/dendrogram_mcrae_domains.pdf"

def get_domain_colors(vocabulary):
	colors = ['r', 'olive', 'g', 'c', 'm', 'y', 'purple']
	label_colors = {x: 'k' for x in vocabulary}
	with open(CONCSTATS, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			if row["Feature"] in DOMAINS:
				label_colors[row["Concept"]] = colors[DOMAINS.index(row["Feature"])]
	return label_colors

def create_X(vocabulary):
	"""
	@inputs
	- vocabulary: set of concepts

	@outputs
	- X: list of lists, where each list represents a vector for a concept
	- labels: concepts, in the same order as its corresponding vector in X
	"""
	X = []
	labels = []

	f = open(INPUT, 'r')
	for line in f:
		word_vec = line.split()
		if word_vec[0] in vocabulary:
			X.append([float(x) for x in word_vec[1:]])
			labels.append(word_vec[0])

	return (X, labels)

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	X, labels = create_X(vocabulary)
	
	Z = hierarchy.linkage(X, method='average', metric='cosine')
	hierarchy.set_link_color_palette(['b']) # or ['m', 'c', 'y', 'k']
	plt.figure(figsize=(70, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dn = hierarchy.dendrogram(Z, labels=labels)
	label_colors = get_domain_colors(vocabulary)
	ax = plt.gca()
	labels = ax.get_xmajorticklabels()
	for l in labels:
		l.set_color(label_colors[l.get_text()])
	savefig(OUTPUT)

if __name__ == '__main__':
	main()
