"""
Outputs McRae Vectors to a txt
so that there is one concept per line,
followed by its vector.
This is so that McRae inputs are the same as
GloVe ones, so I can combine some files into
one.

Example line:
concept 1 0 0 4 0
"""

import csv
import numpy as np
from collections import defaultdict

INPUT = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
#OUTPUT = "./all/mcrae_vectors.txt"

def main():
	# vocab_file = open(VOCAB, 'r')
	# vocabulary = set()
	# for line in vocab_file:
	# 	vocabulary.add(line.strip())

	features = set()
	concepts_feats = defaultdict(list)
	with open(INPUT, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter='\t')
		for row in reader:
			#if row["Concept"] in vocabulary:
			concepts_feats[row["Concept"]].append((row["Feature"], row["Prod_Freq"]))
			features.add(row["Feature"])
	feature_list = sorted(list(features))
	concept_vecs = {word: np.zeros(len(feature_list)) for word in concepts_feats.keys()}
	for concept in concepts_feats:
		feats = concepts_feats[concept]
		for f in feats:
			concept_vecs[concept][feature_list.index(f[0])] = f[1]

	f_per_c = []
	c_per_f = np.zeros(len(features))

	for c in concept_vecs:
		f_per_c.append(np.count_nonzero(concept_vecs[c]))
		for i in range(len(concept_vecs[c])):
			if concept_vecs[c][i] != 0: 
				c_per_f[i] += 1
	print("# features per concept:")
	print("mean\tmedian\tmin\tmax")
	print("%f %f %f %f" % (np.mean(f_per_c), np.median(f_per_c), np.min(f_per_c),
		np.max(f_per_c)))
	print("# concepts per feature")
	print("mean\tmedian\tmin\tmax")
	print("%f %f %f %f" % (np.mean(c_per_f), np.median(c_per_f), np.min(c_per_f),
		np.max(c_per_f)))
	print("num features total")
	print(len(c_per_f))

	# out = open(OUTPUT, 'w')
	# for concept in concept_vecs:
	# 	out.write(concept)
	# 	for ele in concept_vecs[concept]:
	# 		out.write(' ' + str(ele))
	# 	out.write("\n")
	# out.close()

if __name__ == '__main__':
	main()