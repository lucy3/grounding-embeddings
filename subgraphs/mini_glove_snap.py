"""
Using SNAP and GraphViz
http://snap.stanford.edu/snappy/index.html
http://www.graphviz.org

Downside of SNAP: Not easy to assign attributes,
such as weights, to undirected graphs.

This creates a small fruit/veg subgraph based
on cosine similarities of words in glove.6B.300d.txt.

- input: cosine simlarities and vocabulary
- output: graph png

"""

import snap
from collections import defaultdict

INPUT_FILE = "fruitveg_sim_glove.txt"
VOCAB = "vocab_fruitveg.txt"
OUTPUT_GRAPH = "fruitveg_out_glove.png"

def get_cosine_dist():
	d = defaultdict(float)
	word_sim = open(INPUT_FILE, 'r')
	for line in word_sim:
		pair = tuple(line.split()[:2])
		dist = float(line.split()[2])
		d[pair] = dist
	return d

def add_nodes(Graph):
	name_to_id = defaultdict(int)
	vocab_file = open(VOCAB, 'r')
	i = 0
	for line in vocab_file:
		Graph.AddNode(i)
		name_to_id[line.strip()] = i
		i += 1
	return name_to_id

def main():
	cosine_dists = get_cosine_dist()

	Graph = snap.TUNGraph.New()
	name_to_id = add_nodes(Graph)

	for pair in cosine_dists:
		if cosine_dists[pair] > 0.5:
			Graph.AddEdge(name_to_id[pair[0]], name_to_id[pair[1]])

	labels = snap.TIntStrH()
	for name in name_to_id:
		labels[name_to_id[name]] = name
	snap.DrawGViz(Graph, snap.gvlDot, OUTPUT_GRAPH, " ", labels)

if __name__ == '__main__':
	main()