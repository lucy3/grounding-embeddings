"""
Playing around with path distance in WordNet
"""
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

SOURCE = "cslb"
VOCAB = "./all/vocab_%s.txt" % SOURCE

OUTPUT = "./all/sim_%s_wordnetres.txt" % SOURCE

def main():
    brown_ic = wordnet_ic.ic('ic-brown.dat')

    vocab_file = open(VOCAB, 'r')
    vocabulary = set()
    for line in vocab_file:
        vocabulary.add(line.strip())

    word_to_synset = {}
    for concept in vocabulary:
        if concept == 'bluejay':
            senses = wn.synsets('jaybird')
        elif concept == 'rollerskate':
            senses = wn.synsets('roller_skate')
        elif concept == 'wetsuit':
            senses = wn.synsets('wet_suit')
        elif concept == 'yoyo':
            senses = wn.synsets('yo-yo')
        elif concept == 'deckchair':
            senses = wn.synsets('deck_chair')
        else:
            senses = wn.synsets(concept)
        word_to_synset[concept] = senses[0]

    output = open(OUTPUT, 'w')
    words = list(vocabulary)
    num_dists = set()
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            synset1 = word_to_synset[words[i]]
            synset2 = word_to_synset[words[j]]
            # since path similarity isn't commutative
            dist = max(synset1.res_similarity(synset2, brown_ic),
                synset2.res_similarity(synset1, brown_ic))
            num_dists.add(dist)
            output.write(words[i] + ' ' + words[j] + ' ' + str(dist) + '\n')

if __name__ == '__main__':
    main()