"""
Get domains
"""

from nltk.corpus import wordnet as wn
from scipy.cluster import hierarchy
import numpy as np

VOCAB = "./all/vocab.txt"
INPUT = "./all/mcrae_vectors.txt" # McRae
# DOMAINS = '../wndomains/wordnet-domains-3.2-wordnet-3.0.txt'
DOMAINS = './all/lda.txt'

def distance_siblings(Z, labels, threshold):
	"""
	Returns list of lists that are sibling clusters.
	"""
	membership = hierarchy.fcluster(Z, threshold, criterion='maxclust') # maxclust
	sib_clusters = [[] for x in range(max(membership) + 1)]
	for i in range(len(membership)):
		cluster_id = membership[i]
		sib_clusters[cluster_id].append(labels[i])
	return sib_clusters

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

def get_concept_domains(threshold=62): # threshold 62
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	X, labels = create_X(vocabulary)

	Z = hierarchy.linkage(X, method='average', metric='cosine')

	sib_clusters = distance_siblings(Z, labels, threshold)
	new_clusters = []
	new_clust = []
	for cluster in sib_clusters:
		if len(cluster) < 3: # originally 7
			new_clust.extend(cluster)
		else:
			new_clusters.append(cluster)
	new_clusters.append(new_clust)

	concept_domains = {c: [] for c in vocabulary}
	for i, clust in enumerate(new_clusters):
		for c in clust:
			concept_domains[c].append(i)

	return concept_domains

def get_concept_domains_lda():
	"""
	concept: domain string
	"""
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	d_file = open(DOMAINS, 'r')
	concept_domains = {}
	for line in d_file:
		contents = line.split()
		values = [float(x) for x in contents[1:]]
		if contents[0] in vocabulary:
			concept_domains[contents[0]] = [i for i in range(len(values))
			if values[i] == max(values)]
	return concept_domains


def get_concept_domains_old():
	'''
	concept: domain string
	'''
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	# strange edge case of bluejay = blue jay = jaybird in wordnet
	if 'bluejay' in vocabulary:
		vocabulary.remove('bluejay')
		vocabulary.add('jaybird')

	offset_to_domain = {}
	domain_map = open(DOMAINS, 'r')
	for line in domain_map:
		contents = line.split()
		offset_to_domain[contents[0]] = contents[2:] # taking the first domain. each sense can have multiple domains

	concept_domains = {} # concept: [domains]
	for concept in vocabulary:
		senses = wn.synsets(concept)
		offset = str(senses[0].offset()).zfill(8) + '-' + senses[0].pos() # the first sense
		assert senses[0].pos() == 'n' # should at least be a noun
		if offset not in offset_to_domain: # there are some concepts without domain labels
			concept_domains[concept] = ['n/a']
		else:
			concept_domains[concept] = offset_to_domain[offset]
	return concept_domains

def get_domain_concepts():
	concept_domains = get_concept_domains()
	domain_concepts = {} # domain: [concepts]
	for concept in concept_domains:
		domains = concept_domains[concept]
		for d in domains:
			if d in domain_concepts:
				domain_concepts[d].append(concept)
			else:
				domain_concepts[d] = [concept]
	return domain_concepts

if __name__ == '__main__':
	domain_concepts = get_domain_concepts()

	for d in domain_concepts:
		print(d, domain_concepts[d])
