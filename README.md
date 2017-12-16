# Grounding-Embeddings

(This is just a rough summary, will be updated more thoroughly soon.)

## Setup 

You'll likely need: NLTK, sklearn, numpy, matplotlib, and several other packages... 

## Data

- [CSLB property norms](http://csl.psychol.cam.ac.uk/propertynorms/)
- [McRae feature norms](https://link.springer.com/article/10.3758%2FBF03192726#SupplementaryMaterial)
- [GloVe (Wikipedia 2014 + Gigaword 5, Common Crawl)](https://nlp.stanford.edu/projects/glove/)
- [Word2Vec (Google News)](https://code.google.com/archive/p/word2vec/)

To automatically retrieve all of the above except CSLB: 

```bash setup.sh```

## Directory / File naming conventions

The main directory has subgraphs, where we keep everything, and some empty folders that hold data. In subgraphs, all is where we keep our data outputs and intermediates. 


