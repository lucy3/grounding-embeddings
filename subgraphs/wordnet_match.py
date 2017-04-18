"""
Analyze matching between dendrogram and WordNet hierarchy.

1. For a given dendrogram, extract spans at a given depth.
2. Do the same thing for the WordNet hierarchy. (But just
start with the basic WordNet hierarchy.)
3. Element-wise metric: for each element in a span, calculate
p(siblings of span are siblings in WordNet) =
p(sibling WordNet | sibling in span)

NB, both hierarchies will probably need to be "trimmed" - i.e.,
we'll cut the tree off at a certain depth and treat all descendants
below that depth as siblings.

For the output files, the headers have tuples representing
depth/distance. The first value is a parameter for the dendrogram,
and the second for wordnet.
"""

from scipy.cluster import hierarchy
from nltk.corpus import wordnet as wn
import itertools
import operator
import csv

VOCAB = "./all/vocab.txt"

# INPUT = "../glove/glove.6B.300d.txt" # Wikipedia 2014 + Gigaword 5
# OUTPUT = "./all/hier_clust/wordnet_match_wikigiga.txt"

# INPUT = "../glove/glove.840B.300d.txt" # Common Crawl
# OUTPUT = "./all/hier_clust/wordnet_match_cc.txt"

INPUT = "./all/mcrae_vectors.txt" # McRae
OUTPUT = "./all/hier_clust/wordnet_match_mcrae.txt"

def create_X(vocabulary):
	"""
	Copied from hier_clust.py, because INPUT will change and
	don't want to mess with hier_clust.py too much.
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

def distance_siblings(Z, labels, threshold):
	"""
	Returns list of lists that are sibling clusters.
	"""
	membership = hierarchy.fcluster(Z, threshold, criterion='distance')
	sib_clusters = [[] for x in range(max(membership) + 1)]
	for i in range(len(membership)):
		cluster_id = membership[i]
		sib_clusters[cluster_id].append(labels[i])
	return sib_clusters

def depth_siblings(Z, labels, depth):
	"""
	Returns list of lists that are sibling clusters.
	"""
	root = hierarchy.to_tree(Z)
	sib_clusters = []
	node_layer = [root]
	for i in range(depth):
		children = []
		while node_layer:
			node = node_layer.pop()
			if not node.is_leaf():
				children.append(node.right)
				children.append(node.left)
			else:
				children.append(node)
		node_layer = children
	for cluster in node_layer:
		sib_clusters.append(cluster.pre_order(lambda x: labels[x.id]))
	return sib_clusters

def get_ancestors(concept, depth):
	"""
	Gets a layer of ancestors of a concept
	at a certain depth, given that
	entity.n.01 is the root node.
	"""
	ancestors = set()
	senses = wn.synsets(concept)
	for s in senses:
		hypernyms = s.hypernym_paths()
		for path in hypernyms:
			if path[0].name() == 'entity.n.01' and len(path) > depth:
				ancestors.add(path[depth])
	return ancestors

def are_wordnet_siblings(concept1, concept2, depth):
	"""
	Get all hypernym_paths().
	For a specified depth, we find the hyponyms sets of
	'entity.n.01' at this depth in each path for concept1
	and concept2. If these sets intersect, these
	concepts are siblings.
	"""
	ancestors1 = get_ancestors(concept1, depth)
	ancestors2 = get_ancestors(concept2, depth)
	return True if ancestors1 & ancestors2 else False

def write_output(all_probs, params):
	with open(OUTPUT, 'w') as csvfile:
		fieldnames = ["concept"] + ["dendrogram: " +
			str(p[0]) + "; wordnet: " + str(p[1]) for p in params]
		writer = csv.DictWriter(csvfile, delimiter='\t', fieldnames=fieldnames)
		writer.writeheader()
		for concept in all_probs:
			row = {'concept': concept}
			for p in params:
				row["dendrogram: " + str(p[0]) +
					"; wordnet: " + str(p[1])] = all_probs[concept][p]
			writer.writerow(row)

def calculate_probs(dendrogram_param, all_probs, wordnet_depths, labels, sibling_func, Z, params):
	for Dparam in dendrogram_param:
		sibs = sibling_func(Z, labels, Dparam)
		for WNdepth in wordnet_depths:
			both = {x: 0 for x in labels}
			total = {x: 0 for x in labels}
			for span in sibs:
				if len(span) > 2:
					for pair in itertools.combinations(span, r=2):
						total[pair[0]] += 1
						total[pair[1]] += 1
						if are_wordnet_siblings(pair[0], pair[1], WNdepth):
							both[pair[0]] += 1
							both[pair[1]] += 1
			for concept in labels:
				if total[concept] != 0:
					all_probs[concept][(Dparam, WNdepth)] = float(both[concept])/total[concept]
				else:
					all_probs[concept][(Dparam, WNdepth)] = 'n/a'
			params.append((Dparam, WNdepth))
	return (all_probs, params)

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	X, labels = create_X(vocabulary)

	Z = hierarchy.linkage(X, method='average', metric='cosine')

	wordnet_depths = range(6, 9)
	dendrogram_depths = range(6, 9)
	dendrogram_distances = [0.7, 0.8, 0.9]

	all_probs = {x: {} for x in labels}
	params = []
	all_probs, params = calculate_probs(dendrogram_depths, all_probs,
		wordnet_depths, labels, depth_siblings, Z, params)
	all_probs, params = calculate_probs(dendrogram_distances, all_probs,
		wordnet_depths, labels, distance_siblings, Z, params)

	write_output(all_probs, params)

if __name__ == '__main__':
	main()
