from collections import defaultdict
import csv
from pprint import pprint

from gensim import corpora, models
import numpy as np


FEATURES = "../mcrae/CONCS_FEATS_concstats_brm.txt"
OUT = "./all/lda.txt"


def load_concepts_features():
    concepts = defaultdict(list)

    with open(FEATURES, "r") as features_f:
        for i, line in enumerate(features_f):
            if i == 0: continue

            fields = line.strip().split("\t")
            concept_name, feature_name = fields[:2]
            concepts[concept_name].append(feature_name)

    return concepts


def main():
    concepts_features = load_concepts_features()
    concepts = list(sorted(concepts_features.keys()))

    # Prep for LDA.
    # Each "document" is a concept, a bag of features.
    documents = [concepts_features[concept] for concept in concepts]
    dictionary = corpora.Dictionary(documents)
    # Convert to BoW with IDs.
    documents = [dictionary.doc2bow(document) for document in documents]

    # Learn model.
    model = models.LdaModel(documents, id2word=dictionary, num_topics=15)
    pprint(model.print_topics())

    topic_matrix = np.zeros((len(concepts), model.num_topics))
    for i, (concept, document) in enumerate(zip(concepts, documents)):
        for topic_id, weight in model[document]:
            topic_matrix[i, topic_id] = weight

    topic_matrix /= topic_matrix.sum(axis=1, keepdims=True)

    with open(OUT, "w") as out_f:
        headers = ["Concept"] + [str(x) for x in range(model.num_topics)]

        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(headers)
        for concept, topic_vec in zip(concepts, topic_matrix):
            writer.writerow([concept] + [str(x) for x in topic_vec])


if __name__ == '__main__':
    main()
