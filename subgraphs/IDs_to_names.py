"""
Turns modularity communities with IDs of nodes to actual
clusters with names that a human can understand.
"""

from collections import defaultdict

INPUT_NET = "./all/mcrae.net"
INPUT_COMMUN = "./all/mcrae-lol.txt"

def main():
	nodes = defaultdict(str)
	network = open(INPUT_NET, 'r')
	num_nodes = int(network.readline().split()[1])
	for i in range(num_nodes):
		line = network.readline().split()
		nodes[int(line[0])] = line[1]
	network.close()

	communities = open(INPUT_COMMUN, 'r')
	for i in range(7):
		communities.readline()
	for line in communities:
		comm = [nodes[int(x)] for x in line.split()[1:]]
		for item in comm:
			print item,
		print "\n"

if __name__ == '__main__':
	main()