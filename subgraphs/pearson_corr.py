"""
Neighbor-distance correlation between two models
"""

from scipy.stats.stats import pearsonr
from collections import defaultdict
import operator

VOCAB = "./all/vocab.txt"
INPUT_FILE1 = "./all/sim_mcrae.txt"
INPUT_FILE2 = "./all/sim_glove_cc.txt"
OUTPUT_FILE = "./all/corr_mcrae_cc.txt"

def get_cosine_dist(input_file):
	d = defaultdict(float)
	word_sim = open(input_file, 'r')
	for line in word_sim:
		pair = tuple(line.split()[:2])
		dist = float(line.split()[2])
		d[pair] = dist
	return d

def get_neighbor_distance(input_file, vocabulary):
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

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = []
	for line in vocab_file:
		vocabulary.append(line.strip())
	neighbor_dist1 = get_neighbor_distance(INPUT_FILE1, vocabulary)
	neighbor_dist2 = get_neighbor_distance(INPUT_FILE2, vocabulary)
	pearson_co = defaultdict(float)
	for concept in vocabulary:
		if concept != "dunebuggy" and concept != "pipe":
			# dunebuggy not in GloVe data
			# pipe is not supposed to be part of the vocab; inconsistency in McRae
			pearson_co[concept] = pearsonr(neighbor_dist1[concept], neighbor_dist2[concept])[0]
	sorted_pearson = sorted(pearson_co.items(), key=operator.itemgetter(1))

	output = open(OUTPUT_FILE, 'w')
	for pair in sorted_pearson:
		output.write(pair[0] + ' ' + str(pair[1]) + '\n')

if __name__ == '__main__':
	main()
