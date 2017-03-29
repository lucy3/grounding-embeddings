"""
Hierarchical clustering for McRae

main() is the same as hier_clust_glove.py
"""

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from pylab import savefig
import csv
import numpy as np

INPUT = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
OUTPUT = "./all/dendrogram_mcrae.pdf"

def create_X(vocabulary):
	"""
	@inputs
	- vocabulary: set of concepts

	@outputs
	- X: list of lists, where each list represents a concept vector.
	Each vector has the production frequencies of each possible feature
	for a concept. It is with these vectors that McRae produced their cosine 
	matrices. 
	- labels: concepts, in the same order as its corresponding vector in X
	"""
	X = []
	labels = []

	features = set()
	concepts_feats = {word: [] for word in vocabulary}
	with open(INPUT, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			if row["Concept"] in vocabulary:
				concepts_feats[row["Concept"]].append((row["Feature"], row["Prod_Freq"]))
				features.add(row["Feature"])
	feature_list = sorted(list(features))
	concept_vecs = {word: np.zeros(len(feature_list)) for word in vocabulary}
	for concept in concepts_feats:
		feats = concepts_feats[concept]
		for f in feats:
			concept_vecs[concept][feature_list.index(f[0])] = f[1]

	for concept in concept_vecs:
		X.append(concept_vecs[concept])
		labels.append(concept)

	return (X, labels)

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())
	X, labels = create_X(vocabulary)
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