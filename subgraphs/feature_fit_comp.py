"""
Scatter plot of feature_fit comparison of Word2Vec and GloVe
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os.path
from scipy import stats

VOCAB = "mcrae"
SOURCE1 = "word2vec"
SOURCE2 = "wikigiga"
FF1 = "./all/feature_fit/%s/%s/features.txt" % (VOCAB, SOURCE1)
FF2 = "./all/feature_fit/%s/%s/features.txt" % (VOCAB, SOURCE2)
GRAPH_DIR = "./all/feature_fit/%s/" % VOCAB

def read_input(f):
    features = []
    values = []
    file = open(f, 'r')
    for line in file:
        l = line.split()
        features.append(l[0])
        values.append(float(l[3]))
    return (features, values)

def main():
    features1, values1 = read_input(FF1)
    features2, values2 = read_input(FF2)
    assert set(features1) == set(features2)
    xs = [v1 for (f1, v1) in sorted(zip(features1, values1))]
    ys = [v2 for (f2, v2) in sorted(zip(features2, values2))]

    # Jitter points
    xs += np.random.randn(len(xs)) * 0.001
    ys += np.random.randn(len(ys)) * 0.001
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(SOURCE1)
    ax.set_ylabel(SOURCE2)
    ax.scatter(xs, ys, color="Coral")
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    print("slope", slope, "r", r_value)
    plt.plot(xs, slope*xs + intercept, '-', color="red")
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "%s-%s.png" % (SOURCE1, SOURCE2))
    fig.savefig(fig_path)
    plt.close()

if __name__ == '__main__':
    main()