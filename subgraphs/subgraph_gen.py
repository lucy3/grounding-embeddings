"""
This creates a graph based
on cosine similarities of words.

- input: cosine simlarities and vocabulary
- output: graph png or graph statistics txt

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

NUM_EDGES = 600

# uncomment for GloVe fruitveg data
# INPUT_FILE = "./fruitveg/fruitveg_sim_glove.txt"
# VOCAB = "./fruitveg/vocab_fruitveg.txt"
# OUTPUT_GRAPH = "./fruitveg/fruitveg_out_glove" + str(NUM_EDGES) + ".png"
# OUTPUT_STATS = "./fruitveg/fruitveg_glove_stats" + str(NUM_EDGES) + ".txt"

# uncomment for ALL GloVe data
INPUT_FILE = "./all/sim_glove.txt"
VOCAB = "./all/vocab.txt"
OUTPUT_STATS = "./all/glove_stats" + str(NUM_EDGES) + ".txt"
# INPUT_FILE = "./all/sim_glove_cc.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_STATS = "./all/glove_cc_stats" + str(NUM_EDGES) + ".txt"
# INPUT_FILE = "./all/sim_glove_tw.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_STATS = "./all/glove_tw_stats" + str(NUM_EDGES) + ".txt"

# uncomment for McRae fruitveg data
# INPUT_FILE = "./fruitveg/fruitveg_sim_mcrae.txt"
# VOCAB = "./fruitveg/vocab_fruitveg.txt"
# OUTPUT_GRAPH = "./fruitveg/fruitveg_out_mcrae" + str(NUM_EDGES) + ".png"
# OUTPUT_STATS = "./fruitveg/fruitveg_mcrae_stats" + str(NUM_EDGES) + ".txt"

# uncomment for ALL McRae data
# INPUT_FILE = "./all/sim_mcrae.txt"
# VOCAB = "./all/vocab.txt"
# OUTPUT_STATS = "./all/mcrae_stats" + str(NUM_EDGES) + ".txt"

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

def output_graph_stats(g):
	"""
	Clique percolation, along with some other NetworkX statistics
	about the generated graph that may be useful.
	"""
	stat_file = open(OUTPUT_STATS, 'w')
	stat_file.write("NOTE: graph is treated as an unweighted graph" + "\n\n")
	stat_file.write(str(nx.info(g)) + "\n\n")
	stat_file.write("TRANSITIVITY: " + str(nx.transitivity(g)) + "\n\n")
	clust_coeffs = nx.clustering(g)
	stat_file.write("NODES WITH CLUST COEFF = 1: " + "\n")
	for node in clust_coeffs:
		if clust_coeffs[node] == 1.0:
			stat_file.write(node + " " + str(g.neighbors(node)) + "\n")
	stat_file.write("AVG CLUSTERING COEFFICIENT: " +
		str(nx.average_clustering(g)) + "\n\n")
	stat_file.write("DEGREE HISTOGRAM: " + str(nx.degree_histogram(g)) + "\n\n")
	stat_file.write("NODES WITH HIGHEST DEGREE CENTRALITY: " + "\n")
	stat_file.write(str(sorted(nx.degree_centrality(g).items(),
		key=operator.itemgetter(1), reverse=True)[:5]) + "\n\n")
	stat_file.write("4-CLIQUE COMMUNITIES (clique percolation): " + "\n")
	for clique in nx.k_clique_communities(g, 4):
		stat_file.write(str(clique) + "\n")
	stat_file.write("\nMAXIMAL CLIQUES: " + "\n")
	for clique in nx.find_cliques(g):
		if len(clique) >= 3:
			stat_file.write(str(clique) + "\n")

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

	# draw(g, filename=OUTPUT_GRAPH)
	output_graph_stats(g)

if __name__ == '__main__':
	main()