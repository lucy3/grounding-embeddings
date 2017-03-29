"""
Using this community detection tool:
http://deim.urv.cat/~sergio.gomez/radatools.php

This generates a pajek .net file representing our network
so that the above tool can work.

This is a undirected, weighted, signed, complete network.

Usage of output .net:
Communities_Detection.exe P WS trfr 1 net_name.net net_name-lol.txt
"""

import networkx as nx
from collections import defaultdict

# uncomment for ALL GloVe data
# INPUT_FILE = "./all/sim_glove.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_NET = "./all/modularity/glove.net"
# INPUT_FILE = "./all/sim_glove_cc.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_NET = "./all/modularity/glove_cc.net"
INPUT_FILE = "./all/sim_glove_tw.txt"
VOCAB = "./all/vocab.txt"
OUTPUT_NET = "./all/modularity/glove_tw.net"

# uncomment for ALL McRae data
# INPUT_FILE = "./all/sim_mcrae.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_NET = "./all/modularity/mcrae.net"

def get_cosine_dist():
	"""
	@output:
	- d: {(concept1, concept2) tuple : distance as a float}
	"""
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
		g.add_edge(pair[0], pair[1])
		g[pair[0]][pair[1]]["weight"] = cosine_dists[pair]

	nx.write_pajek(g, OUTPUT_NET)

	vocab_file.close()

if __name__ == '__main__':
	main()
