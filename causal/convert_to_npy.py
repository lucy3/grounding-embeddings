"""
Convert the GloVe co-occurrence data to a numpy matrix.
"""

from argparse import ArgumentParser
import struct

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from tqdm import trange


p = ArgumentParser()
p.add_argument("--vocab_file", required=True)
p.add_argument("--cooccur_file", required=True)
p.add_argument("--out_file", default="cooccur.npz")

args = p.parse_args()


vocab = []
with open(args.vocab_file, "r") as vocab_f:
    for line in vocab_f:
        vocab.append(line.strip().split()[0])

print("Read a vocabulary with %i tokens from %s." % (len(vocab), args.vocab_file))


v2i = {v: i for i, v in enumerate(vocab)}
with open(args.cooccur_file, "rb") as cooccur_f:
    data = []
    i, j = [], []
    for word1, word2, val in struct.iter_unpack("iid", cooccur_f.read()):
        data.append(val)
        i.append(word1 - 1)
        j.append(word2 - 1)
        # if crec.word1 - 1 == v2i["foot"]:
        #     print("foot", vocab[crec.word2 - 1])

    matrix = coo_matrix((data, (i, j)), shape=(len(vocab), len(vocab)))


def save_coo(filename, coo):
    np.savez(filename, row=coo.row, col=coo.col, data=coo.data, shape=coo.shape)

save_coo(args.out_file, matrix)
