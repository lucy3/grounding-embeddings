"""
Using NetworkX

This creates a small fruit/veg subgraph based
on cosine similarities of words. 

- input: cosine simlarities and vocabulary
- output: graph png

"""

import networkx as nx
from collections import defaultdict
from nxpd import draw

# # uncomment for GloVe data
# INPUT_FILE = "fruitveg_sim_glove.txt"
# VOCAB = "vocab_fruitveg.txt"
# OUTPUT_GRAPH = "fruitveg_out_glove.png"

# uncomment for McRae data
INPUT_FILE = "fruitveg_sim_mcrae.txt"
VOCAB = "vocab_fruitveg.txt"
OUTPUT_GRAPH = "fruitveg_out_mcrae.png"

def get_cosine_dist():
	d = defaultdict(float)
	word_sim = open(INPUT_FILE, 'r')
	for line in word_sim:
		pair = tuple(line.split()[:2])
		dist = float(line.split()[2])
		d[pair] = dist
	return d

def main():
	cosine_dists = get_cosine_dist()

	g = nx.Graph()
	vocab_file = open(VOCAB, 'r')
	for line in vocab_file:
		g.add_node(line.strip())
	for pair in cosine_dists:
		if cosine_dists[pair] > 0.5:
			g.add_edge(pair[0], pair[1])
			# I would use "weight" but that
			# makes the visualization look funny
			# "weight" will be helpful for
			# running some NetworkX algorithms
			g[pair[0]][pair[1]]["w"] = cosine_dists[pair]

	draw(g, filename=OUTPUT_GRAPH)

if __name__ == '__main__':
	main()