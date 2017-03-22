"""
This computes cosine distances based on GloVe data. 

vocab_fruitveg.txt is hand-picked list of fruits/vegs
from the McRae concept dataset.

vocab.txt is a list of concepts from McRae without underscores
(underscores differentiate multiple meanings for one concept).

- input: vocabulary, glove data
- output: cosine similarities in a txt
"""

from collections import defaultdict
from scipy import spatial

# GLOVE_INPUT = "../glove/glove.6B.300d.txt" # Wikipedia 2014 + Gigaword 5
# GLOVE_INPUT = "../glove/glove.840B.300d.txt" # Common Crawl
GLOVE_INPUT = "../glove/glove.twitter.27B.200d.txt" # Twitter
# VOCAB = "./fruitveg/vocab_fruitveg.txt"
# OUTPUT = "./fruitveg/fruitveg_sim_glove.txt"
VOCAB = "./all/vocab.txt"
# OUTPUT = "./all/sim_glove.txt"
# OUTPUT = "./all/sim_glove_cc.txt"
OUTPUT = "./all/sim_glove_tw.txt"

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())
		
	f = open(GLOVE_INPUT, 'r')
	vectors = defaultdict(list)
	for line in f:
		word_vec = line.split()
		if word_vec[0] in vocabulary:
			vectors[word_vec[0]] = [float(x) for x in word_vec[1:]]

	output = open(OUTPUT, 'w')
	words = vectors.keys()
	for i in range(len(words)):
		for j in range(i+1, len(words)):
			dist = 1- spatial.distance.cosine(vectors[words[i]], vectors[words[j]])
			output.write(words[i] + ' ' + words[j] + ' ' + str(dist) + '\n')

if __name__ == '__main__':
	main()
