"""
Calculate cosine distance of word2vec
"""

from gensim.models.keyedvectors import KeyedVectors

VOCAB = "./all/vocab_cslb.txt"
INPUT = '../word2vec/GoogleNews-vectors-negative300.bin'
OUTPUT = "./all/sim_cslb_word2vec.txt"

def add_American(vocabulary):
	vocabulary.remove("whisky")
	vocabulary.add("whiskey")
	vocabulary.remove("catalogue")
	vocabulary.add("catalog")
	vocabulary.add("armor")
	vocabulary.remove("armour")
	vocabulary.remove("plough")
	vocabulary.add("plow")
	vocabulary.remove("tyre")
	vocabulary.add("tire")
	vocabulary.remove("aeroplane")
	vocabulary.add("airplane")
	vocabulary.remove("pyjamas")
	vocabulary.add("pajamas")
	vocabulary.remove("doughnut")
	vocabulary.add("donut")
	vocabulary.remove("axe")
	vocabulary.add("ax")
	return vocabulary

def main():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	# British English
	vocabulary = add_American(vocabulary)

	model = KeyedVectors.load_word2vec_format(INPUT, binary=True)

	exists = set()
	for word in vocabulary:
		if word in model.vocab:
			exists.add(word)
	print vocabulary-exists

	output = open(OUTPUT, 'w')
	words = list(vocabulary)
	B_words = list(words)
	for i in range(len(B_words)):
		if B_words[i] == "whiskey":
			B_words[i] = "whisky"
		if B_words[i] == "catalog":
			B_words[i] = "catalogue"
		if B_words[i] == "armor":
			B_words[i] = "armour"
		if B_words[i] == "plow":
			B_words[i] = "plough"
		if B_words[i] == "tire":
			B_words[i] = "tyre"
		if B_words[i] == "airplane":
			B_words[i] = "aeroplane"
		if B_words[i] == "pajamas":
			B_words[i] = "pyjamas"
		if B_words[i] == "donut":
			B_words[i] = "doughnut"
		if B_words[i] == "ax":
			B_words[i] = "axe"
	for i in range(len(words)):
		for j in range(i+1, len(words)):
			dist = model.similarity(words[i], words[j])
			output.write(B_words[i] + ' ' + B_words[j] + ' ' + str(dist) + '\n')

if __name__ == '__main__':
	main()