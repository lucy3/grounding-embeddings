"""
This outputs cosine distances for a subset of fruits/vegs
based on McRae data.

mcrae_fruitveg.txt is hand-picked list of fruits/vegs
from the McRae concept dataset.

- input: vocabulary, mcrae matrices
- output: txt of cosine similarities
"""

MCRAE_INPUTS = ["../mcrae/cos_matrix_brm_IFR_1-200.txt",
	"../mcrae/cos_matrix_brm_IFR_201-400.txt",
	"../mcrae/cos_matrix_brm_IFR_401-541.txt"]
VOCAB = "./fruitveg/vocab_fruitveg.txt"
OUTPUT = "./fruitveg/fruitveg_sim_mcrae.txt"

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	output = open(OUTPUT, 'w')

	done = set()

	for input_file in MCRAE_INPUTS:
		f = open(input_file, 'r')
		words = f.readline().split()
		for line_number, line in enumerate(f, 1):
			contents = line.split()
			for i in range(1, len(contents)):
				if contents[0] in vocabulary and words[i] in vocabulary \
				and (words[i], contents[0]) not in done and contents[0] != words[i] \
				and float(contents[i]) > 0:
					output.write(contents[0] + ' ' + words[i] + ' ' + contents[i] + '\n')
					done.add((contents[0], words[i]))

if __name__ == '__main__':
	main()