"""
Scatter plot of feature_fit comparison of Word2Vec and GloVe
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os.path
from scipy import stats

VOCAB = "cslb"
SOURCE1 = "word2vec"
SOURCE2 = "cc"
FF1 = "./all/feature_fit/%s/%s/features.txt" % (VOCAB, SOURCE1)
FF2 = "./all/feature_fit/%s/%s/features.txt" % (VOCAB, SOURCE2)
GRAPH_DIR = "./all/feature_fit/%s/" % VOCAB
if SOURCE2 == "cc":
    FORMAL_SOURCE2 = "GloVe (Common Crawl)"

def read_input(f):
    features = []
    values = []
    cats = []
    file = open(f, 'r')
    for line in file:
        l = line.split('\t')
        features.append(l[0])
        values.append(float(l[3]))
        cats.append(l[1])
    return (features, values, cats)

def main():
    features1, values1, cats1 = read_input(FF1)
    features2, values2, cats2 = read_input(FF2)
    assert set(features1) == set(features2)
    xs = [v1 for (f1, v1) in sorted(zip(features1, values1))]
    ys = [v2 for (f2, v2) in sorted(zip(features2, values2))]
    zs = [c1 for (f1, c1) in sorted(zip(features1, cats1))]
    sort_feats = sorted(set(zs))
    # colors_dict = dict(zip(sort_feats, ["Plum", "DarkOrange",
    #     "SkyBlue", "YellowGreen", "IndianRed"]))
    colors_dict = dict(zip(sort_feats, ["LightBlue", "LightBlue",
        "LightBlue", "LightBlue", "LightBlue"]))
    print(colors_dict)
    colors = [colors_dict[zs[c]] for c in range(len(zs))]

    # Jitter points
    xs += np.random.randn(len(xs)) * 0.001
    ys += np.random.randn(len(ys)) * 0.001
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111)
    axes = plt.gca()
    axes.set_xlim([-0.05,105])
    axes.set_ylim([-0.05,105])
    ax.set_xlabel(SOURCE1 + " feature fit")
    ax.set_ylabel(FORMAL_SOURCE2 + " feature fit")
    ax.scatter(xs*100, ys*100, color=colors, linewidth=0.0)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    print("slope", slope, "r", r_value)
    plt.plot(xs*100, slope*xs*100 + intercept, '-', color="DarkBlue", linewidth=2.0)
    plt.tight_layout()
    fig_path = os.path.join(GRAPH_DIR, "%s-%s.eps" % (SOURCE1, SOURCE2))
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # for i in range(len(sort_feats)):
    #     fig = plt.figure()
    #     plt.title(sort_feats[i])
    #     ax = fig.add_subplot(111)
    #     ax.set_xlabel(SOURCE1 + " feature fit")
    #     ax.set_ylabel(FORMAL_SOURCE2 + " feature fit")
    #     for j in range(len(zs)):
    #         if zs[j] == sort_feats[i]:
    #             ax.scatter(xs[j], ys[j], c=colors_dict[sort_feats[i]])
    #     plt.tight_layout()
    #     fig_path = os.path.join(GRAPH_DIR, "%s-%s-%s.eps" % (sort_feats[i], SOURCE1, SOURCE2))
    #     fig.savefig(fig_path)
    #     plt.close()

if __name__ == '__main__':
    main()