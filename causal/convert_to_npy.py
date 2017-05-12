"""
Convert the GloVe co-occurrence data to a numpy matrix.
"""

from argparse import ArgumentParser
from ctypes import *

import numpy as np
from scipy.sparse import lil_matrix
import tables as tb
from tqdm import trange


p = ArgumentParser()
p.add_argument("--vocab_file", required=True)
p.add_argument("--cooccur_file", required=True)
p.add_argument("--out_file", default="cooccur.npz")

args = p.parse_args()


class CREC(Structure):
    _fields_ = [("word1", c_int),
                ("word2", c_int),
                ("val", c_double)]

    def __str__(self):
        return "(%i, %i, %f)" % (self.word1, self.word2, self.val)

    def __repr__(self):
        return str(self)


vocab = []
with open(args.vocab_file, "r") as vocab_f:
    for line in vocab_f:
        vocab.append(line.strip().split()[0])

print("Read a vocabulary with %i tokens from %s." % (len(vocab), args.vocab_file))


print("Initializing vocab * vocab sparse co-occurrence matrix.")
matrix = lil_matrix((len(vocab), len(vocab)), dtype=np.float32)
print("Done")


v2i = {v: i for i, v in enumerate(vocab)}
with open(args.cooccur_file, "rb") as cooccur_f:
    crec = CREC()

    while cooccur_f.readinto(crec) == sizeof(crec):
        if crec.word1 - 1 == v2i["foot"]:
            print("foot", vocab[crec.word2 - 1])
        matrix[crec.word1 - 1, crec.word2 - 1] = crec.val


def save_lil(filename, lil):
    coo = lil.tocoo()
    np.savez(filename, row=coo.row, col=coo.col, data=coo.data, shape=coo.shape)

save_lil(args.out_file, matrix)
