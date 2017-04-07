"""
Automatically compute rank correlation between
dendrogram and WordNet between sources (GloVe vs McRae)
"""

import csv
import scipy.stats as stats
import operator

INPUT1 = "./all/hier_clust/wordnet_match_cc.txt"
#INPUT1 = "./all/hier_clust/wordnet_match_wikigiga.txt"
INPUT2 = "./all/hier_clust/wordnet_match_mcrae.txt"

def read_inputs(input_file):
	input_dict = {}
	with open(input_file, 'rU') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			keys = row.keys()
			keys.remove("concept")
			input_dict[row["concept"]] = {k: row[k] for k in keys}
	return input_dict

def get_input_cols(input_dict):
	concepts = sorted(input_dict.keys())
	input_cols = {}
	for i in range(len(concepts)):
		c = concepts[i]
		row = input_dict[c]
		for param in row:
			if row[param] == "n/a":
				row[param] = 0
			if param not in input_cols:
				input_cols[param] = [float(row[param])]
			else:
				input_cols[param].append(float(row[param]))
	return input_cols

def main():
	input1_dict = read_inputs(INPUT1)
	input2_dict = read_inputs(INPUT2)
	input_cols1 = get_input_cols(input1_dict)
	input_cols2 = get_input_cols(input2_dict)
	assert input_cols1.keys() == input_cols2.keys()
	results_rho = {}
	results_p = {}
	for param in input_cols1:
		res1, res2 = stats.spearmanr(input_cols1[param], input_cols2[param])
		results_rho[param] = res1
		results_p[param] = res2
	sorted_rho = sorted(results_rho.items(), key=operator.itemgetter(1))
	print "parameters", "\t", "rho", "p-value"
	for i in range(len(sorted_rho)):
		print sorted_rho[i][0], "\t", sorted_rho[i][1], results_p[sorted_rho[i][0]]


if __name__ == '__main__':
	main()
