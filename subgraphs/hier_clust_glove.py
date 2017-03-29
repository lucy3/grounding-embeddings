"""
Hierarchical clustering for GloVe
"""

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from pylab import savefig

# INPUT = "../glove/glove.6B.300d.txt" # Wikipedia 2014 + Gigaword 5
INPUT = "../glove/glove.840B.300d.txt" # Common Crawl
VOCAB = "./all/vocab.txt"
# OUTPUT = "./all/dendrogram_wikigiga.pdf"
OUTPUT = "./all/dendrogram_cc.pdf"

def main():
	X = []
	labels = []

	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	f = open(INPUT, 'r')
	for line in f:
		word_vec = line.split()
		if word_vec[0] in vocabulary:
			X.append([float(x) for x in word_vec[1:]])
			labels.append(word_vec[0])
	
	Z = hierarchy.linkage(X, method='average', metric='cosine')
	hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
	plt.figure(figsize=(70, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dn = hierarchy.dendrogram(Z, labels=labels)
	savefig(OUTPUT)

if __name__ == '__main__':
	main()
