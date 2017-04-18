from collections import defaultdict
import csv
from pprint import pprint

from gensim import corpora, models, matutils
import numpy as np
from scipy.spatial import distance


FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
VOCAB = "./all/vocab.txt"
OUT = "./all/mcrae_vectors.txt"
OUT_DISTANCES = "./all/sim_mcrae.txt"


def load_concepts_features():
    concepts = defaultdict(list)

    with open(VOCAB, "r") as vocab_f:
        vocab = [line.strip() for line in vocab_f]

    with open(FEATURES, "r") as features_f:
        for i, line in enumerate(features_f):
            if i == 0: continue

            fields = line.strip().split("\t")
            concept_name, feature_name = fields[:2]
            if concept_name in vocab:
                concepts[concept_name].append(feature_name)

    return concepts


def report_closest(concepts, matrix, sample_concepts, n=50):
#    matrix = matrix[sample_concepts]
    print(matrix.shape)
    distances = distance.pdist(matrix, metric="cosine")
    distances_square = distance.squareform(distances)
    print(distances_square.shape)
    np.fill_diagonal(distances_square, np.inf)

    topn = np.argsort(distances_square.flatten())[:n]
    for idx in topn:
        i, j = np.unravel_index(idx, distances_square.shape)
        print("%20s\t%20s\t%5f" % (concepts[i], concepts[j], distances_square[i, j]))


def main():
    concepts_features = load_concepts_features()
    concepts = list(sorted(concepts_features.keys()))

    # Prep for LSA.
    # Each "document" is a concept, a bag of features.
    #
    # TODO: perhaps replicate according to production frequency? log(production
    # frequency) ?
    documents = [concepts_features[concept] for concept in concepts]
    dictionary = corpora.Dictionary(documents)
    # Convert to BoW with IDs.
    documents = [dictionary.doc2bow(document) for document in documents]

    raw_matrix = matutils.corpus2dense(documents, dictionary.num_nnz).T
    # raw_matrix = np.zeros((len(concepts), len(dictionary.token2id)))
    # for i, document in enumerate(documents):
    #     for term, count in document:
    #         raw_matrix[document, term] = count

    report_closest(concepts, raw_matrix, None)

    # Learn model.
    model = models.LsiModel(documents, id2word=dictionary, num_topics=20)
    pprint(model.print_topics())

    topic_matrix = np.zeros((len(concepts), model.num_topics))
    for i, (concept, document) in enumerate(zip(concepts, documents)):
        for topic_id, weight in model[document]:
            topic_matrix[i, topic_id] = weight

    report_closest(concepts, topic_matrix, None)

    with open(OUT, "w") as out_f:
        headers = ["Concept"] + [str(x) for x in range(model.num_topics)]

        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(headers)
        for concept, topic_vec in zip(concepts, topic_matrix):
            writer.writerow([concept] + [str(x) for x in topic_vec])

    distances = distance.pdist(topic_matrix, metric="cosine")
    distances_square = distance.squareform(distances)
    with open(OUT_DISTANCES, "w") as distances_f:
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i == j: continue
                distances_f.write("%s\t%s\t%8f\n"
                        % (concepts[i], concepts[j],
                           np.clip(1 - distances_square[i, j], -1, 1)))


if __name__ == '__main__':
    main()
