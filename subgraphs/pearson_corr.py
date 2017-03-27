"""
Neighbor-distance correlation between two models

Also outputs the following stats:
KF frequency (KF_freq)
BNC frequency (BNC_freq)
number of features (num_feats_tax)
familiarity (familiarity)
total # of features produced by participants (tot_num_feats)
"""

from scipy.stats.stats import pearsonr
from collections import defaultdict
import operator
import csv

VOCAB = "./all/vocab.txt"
INPUT_FILE1 = "./all/sim_mcrae.txt"
INPUT_FILE2 = "./all/sim_glove_tw.txt"
OUTPUT_FILE = "./all/corr_mcrae_tw.txt"
CONC_BRM = "../mcrae/CONCS_brm.txt"
CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"

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

def get_mcrae_freq():
	concept_stats = defaultdict(list)
	prod_freqs = defaultdict(int)
	with open(CONCSTATS, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			prod_freqs[row["Concept"]] += int(row["Prod_Freq"])

	with open(CONC_BRM, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			concept_stats[row["Concept"]] = row["BNC"] + '\t' + row["Num_Feats_Tax"] + '\t' + \
				row["Familiarity"] + '\t' + str(prod_freqs[row["Concept"]])
	return concept_stats

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = []
	for line in vocab_file:
		vocabulary.append(line.strip())
	neighbor_dist1 = get_neighbor_distance(INPUT_FILE1, vocabulary)
	neighbor_dist2 = get_neighbor_distance(INPUT_FILE2, vocabulary)
	pearson_co = defaultdict(float)
	for concept in vocabulary:
			pearson_co[concept] = pearsonr(neighbor_dist1[concept], neighbor_dist2[concept])[0]
	sorted_pearson = sorted(pearson_co.items(), key=operator.itemgetter(1))

	concept_stats = get_mcrae_freq()

	output = open(OUTPUT_FILE, 'w')
	output.write('Concept\tcorrelation\tBNC_freq\t' +
		'num_feats_tax\tfamiliarity\ttot_num_feats\n')
	for pair in sorted_pearson:
		output.write(pair[0] + '\t' + str(pair[1]) + '\t' + concept_stats[pair[0]] + '\n')

if __name__ == '__main__':
	main()
