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
from nltk.corpus import wordnet as wn

VOCAB = "./all/vocab.txt"
INPUT_FILE1 = "./all/sim_mcrae.txt"
INPUT_FILE2 = "./all/sim_glove.txt"
OUTPUT_FILE = "./all/corr_mcrae_wikigiga.txt"
CONC_BRM = "../mcrae/CONCS_brm.txt"
CONCSTATS = "../mcrae/CONCS_FEATS_concstats_brm.txt"
DOMAINS = set(["a_bird", "a_fish", "a_fruit", "a_mammal", \
	"a_musical_instrument", "a_tool", "a_vegetable", "an_animal"])

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

def get_mcrae_freq(pearson_co):
	concept_stats = defaultdict(list)
	prod_freqs = defaultdict(int)
	sum_in_domain = defaultdict(float)
	count_in_domain = defaultdict(int)
	with open(CONCSTATS, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			prod_freqs[row["Concept"]] += int(row["Prod_Freq"])
			if row["Feature"] in DOMAINS:
				sum_in_domain[row["Feature"]] += pearson_co[row["Concept"]]
				count_in_domain[row["Feature"]] += 1

	with open(CONC_BRM, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			concept_stats[row["Concept"]] = row["BNC"] + '\t' + row["Num_Feats_Tax"] + '\t' + \
				row["Familiarity"] + '\t' + str(prod_freqs[row["Concept"]])
	average_in_domain = defaultdict(float)
	for key in sum_in_domain:
		average_in_domain[key] = sum_in_domain[key]/count_in_domain[key]
	return (concept_stats, average_in_domain)

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

	concept_stats, average_in_domain = get_mcrae_freq(pearson_co)

	output = open(OUTPUT_FILE, 'w')
	output.write('Concept\tcorrelation\tBNC_freq\t' +
		'num_feats_tax\tfamiliarity\ttot_num_feats\tpolysemy\n')
	for pair in sorted_pearson:
		output.write(pair[0] + '\t' + str(pair[1]) + '\t' + concept_stats[pair[0]] + '\t'
			+ str(len(wn.synsets(pair[0]))) + '\n')

	for tax_feature in average_in_domain:
		output.write("\n" + tax_feature + "\t" + str(average_in_domain[tax_feature]) + "\n")

	output.close()

if __name__ == '__main__':
	main()
