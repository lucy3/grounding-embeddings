"""
Using NetworkX

This creates a small fruit/veg subgraph based
on cosine similarities of words.

- input: cosine simlarities and vocabulary
- output: graph png

Note: 
I would use "weight" as an edge attribute but that
makes the visualization look funny.
"weight" will be helpful for
running some NetworkX algorithms.
"""

import networkx as nx
from collections import defaultdict
from nxpd import draw
import operator

NUM_EDGES = 200

# uncomment for GloVe data
INPUT_FILE = "./fruitveg/fruitveg_sim_glove.txt"
VOCAB = "./fruitveg/vocab_fruitveg.txt"
OUTPUT_GRAPH = "./fruitveg/fruitveg_out_glove" + str(NUM_EDGES) + ".png"

# uncomment for McRae data
# INPUT_FILE = "./fruitveg/fruitveg_sim_mcrae.txt"
# VOCAB = "./fruitveg/vocab_fruitveg.txt"
# OUTPUT_GRAPH = "./fruitveg/fruitveg_out_mcrae" + str(NUM_EDGES) + ".png"

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
	sorted_cosine_dists = sorted(cosine_dists.items(),
		key=operator.itemgetter(1), reverse=True)
	for i in range(NUM_EDGES):
		tup = sorted_cosine_dists[i]
		pair = tup[0]
		g.add_edge(pair[0], pair[1])
		g[pair[0]][pair[1]]["w"] = tup[1]

	draw(g, filename=OUTPUT_GRAPH)

if __name__ == '__main__':
	main()