# Graphs-Embeddings

(TODO: Set up project in a virtual env to make project easier to clone and setup. Was facing some technical difficulties so for now, this is less elegantly organized than it could be.) 

## Setup 

1. Download and install the following: [NetworkX](https://pypi.python.org/pypi/networkx/) and [nxpd](https://github.com/chebee7i/nxpd). It is intended that NetworkX is used for graph analysis moving forward. [Snap.py.](http://snap.stanford.edu/snappy/index.html) and [GraphViz](http://www.graphviz.org) were also briefly used. 

2. Download Wikipedia 2014 + Gigaword 5, Common Crawl (840B tokens), and Twitter pre-trained vectors from [GloVe](http://nlp.stanford.edu/projects/glove/) and place in a folder called "glove" in main directory. 

3. Download McRae Norms from [psychonomic](www.psychonomic.org/archive) and place in a folder called "mcrae" in main directory. 

4. Download Communities_Detection.exe from [Radatools](http://deim.urv.cat/~sergio.gomez/radatools.php). 

## Directory / File naming conventions

**subgraphs**
	- mini_DATA_dist.py: outputs cosine distances for some vocabulary based on some DATA source
	- subgraph_gen.py: takes cosine distances and creates a graph and outputs some metrics
	- *fruitveg*: fruit/veg subgraph for GloVe (Wikipedia 2014 + Gigaword 5) and McRae
	- *all*: (almost) all McRae concepts for GloVe (Wikipedia 2014 + Gigaword 5) and McRae
