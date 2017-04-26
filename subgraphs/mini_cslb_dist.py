"""
CSLB: get similarities with LSA'd vectors
"""
import numpy as np

GLOVES = ["../glove/glove.6B.300d.txt", "../glove/glove.840B.300d.txt"]
FEATURE_CONCEPTS = "../cslb/feature_matrix.dat"
VOCAB = "./all/vocab_cslb.txt"

def create_vocab_file(vocab):
	print(len(vocab))
	for g in GLOVES:
		g_vocab = set()
		with open(g, 'r') as glove_file:
			for line in glove_file:
				g_vocab.add(line.split()[0])
		vocab -= (vocab - (g_vocab & vocab))
	print(len(vocab)) # should be 5 less

	vocab_file = open(VOCAB, 'w')
	for word in vocab:
		vocab_file.write(word + "\n")

def process_feature_concepts():
	# vocab = set()
	f_per_c = []
	with open(FEATURE_CONCEPTS, 'r') as file:
		features = file.readline().split()[1:]
		c_per_f = np.zeros(len(features))
		for line in file:
			l = line.split()
			concept = l[0]
			features = np.array([float(x) for x in l[1:]])
			f_per_c.append(np.count_nonzero(features))
			for i in range(len(features)):
				if features[i] != 0:
					c_per_f[i] += 1
			# if "_" not in concept:
			# 	vocab.add(concept)
	print("# features per concept:")
	print("mean\tmedian\tmin\tmax")
	print("%f %f %f %f" % (np.mean(f_per_c), np.median(f_per_c), np.min(f_per_c),
		np.max(f_per_c)))
	print("# concepts per feature")
	print("mean\tmedian\tmin\tmax")
	print("%f %f %f %f" % (np.mean(c_per_f), np.median(c_per_f), np.min(c_per_f),
		np.max(c_per_f)))
	print("num features with >= 5 concepts")
	print(sum(i >= 5 for i in c_per_f))
	print("num features total")
	print(len(c_per_f))

	#create_vocab_file(vocab)


def preliminaries():
	pass


def main():
	process_feature_concepts()
	preliminaries()

if __name__ == '__main__':
	main()
