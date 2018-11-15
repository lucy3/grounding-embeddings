# Grounding-Embeddings

(This is just a rough summary, will be updated more thoroughly.)

## Are distributional representations ready for the real world? Evaluating word vectors for grounded perceptual meaning

Distributional word representation methods exploit word co-occurrences to build compact vector encodings of words. While these representations enjoy widespread use in modern natural language processing, it is unclear whether they accurately encode all necessary facets of conceptual meaning. In this paper, we evaluate how well these representations can predict perceptual and conceptual features of concrete concepts, drawing on two semantic norm datasets sourced from human participants. We find that several standard word representations fail to encode many salient perceptual features of concepts, and show that these deficits correlate with word-word similarity prediction errors. Our analyses provide motivation for grounded and embodied language learning approaches, which may help to remedy these deficits.

[Link to paper](http://aclweb.org/anthology/W17-2810).

## Setup 

You'll need: NLTK, sklearn, numpy, matplotlib, and several other packages.

## Data

- [CSLB property norms](http://csl.psychol.cam.ac.uk/propertynorms/)
- [McRae feature norms](https://link.springer.com/article/10.3758%2FBF03192726#SupplementaryMaterial)
- [GloVe (Wikipedia 2014 + Gigaword 5, Common Crawl)](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec (Google News)](https://code.google.com/archive/p/word2vec/)

To automatically retrieve all of the above except CSLB: 

```bash setup.sh```

## Directory 

The main directory has `subgraphs`, where we keep our code, data outputs, and intermediates. The folders `cslb`, `mcrae`, `glove`, and `word2vec` are empty but should store the data mentioned above. 

Our code is most compatible with Python 3. 

The script `feature_fit.py` computes feature fit scores for words, as described in our paper. Note that the GloVe inputs are in word2vec format. These files should be the same as downloaded GloVe files except it includes the number of vectors and its dimension. So, the top of `glove.6B.300d.w2v.txt` has an extra line with "400000 300" and `glove.840B.300d.w2v.txt` has an extra line with "2196017 300".


