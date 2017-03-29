"""
Hierarchical clustering
"""

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from pylab import savefig

VOCAB = "./all/vocab.txt"

# INPUT = "../glove/glove.6B.300d.txt" # Wikipedia 2014 + Gigaword 5
# OUTPUT = "./all/dendrogram_wikigiga_domains.pdf"

# INPUT = "../glove/glove.840B.300d.txt" # Common Crawl
# OUTPUT = "./all/dendrogram_cc_domains.pdf"

INPUT = "./all/mcrae_vectors.txt" # McRae
OUTPUT = "./all/dendrogram_mcrae.pdf"

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
	hierarchy.set_link_color_palette(['m', 'c', 'y', 'k']) # or ['m', 'c', 'y', 'k']
	plt.figure(figsize=(70, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dn = hierarchy.dendrogram(Z, labels=labels)
	# label_colors = {'a': 'r', 'b': 'g', 'c': 'b', 'd': 'm'}
	# ax = plt.gca()
	# labels = ax.get_xmajorticklabels()
	# for l in labels:
	# 	l.set_color(label_colors[l.get_text()])
	savefig(OUTPUT)

if __name__ == '__main__':
	main()
