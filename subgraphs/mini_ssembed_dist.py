"""
This computes Tantimoto distance based on SensEmbed vectors

(They also have T* which includes a graph vicinity
factor that needs to be tuned so I'm just using T for now)
"""

INPUT = "../sensembed/babelfy_vectors"
VOCAB = "./all/vocab.txt"
OUTPUT = "./all/sim_ssembed.txt"

from collections import defaultdict
import numpy as np

def main():
    vocab_file = open(VOCAB, 'r')
    vocabulary = set()
    for line in vocab_file:
        vocabulary.add(line.strip())

    f = open(INPUT, 'r')
    vectors = defaultdict(list)
    for line in f:
        l = line.split()
        if l[0] in vocabulary:
            vectors[l[0]] = np.array([float(x) for x in l[1:]])

    output = open(OUTPUT, 'w')
    words = vectors.keys()
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            vector1 = vectors[words[i]]
            vector2 = vectors[words[j]]
            dist = (vector1.dot(vector2)) / (vector1.dot(vector1) +
                vector2.dot(vector2) - vector1.dot(vector2))
            output.write(words[i] + ' ' + words[j] + ' ' + str(dist) + '\n')

    f.close()

if __name__ == '__main__':
    main()