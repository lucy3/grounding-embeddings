"""
Get domains
"""

from nltk.corpus import wordnet as wn
from scipy.cluster import hierarchy
import numpy as np

VOCAB = "./all/vocab.txt"
INPUT = "./all/mcrae_vectors.txt" # McRae
# DOMAINS = '../wndomains/wordnet-domains-3.2-wordnet-3.0.txt'
DOMAINS = './all/lda.txt'

def distance_siblings(Z, labels, threshold):
	"""
	Returns list of lists that are sibling clusters.
	"""
	membership = hierarchy.fcluster(Z, threshold, criterion='maxclust')
	sib_clusters = [[] for x in range(max(membership) + 1)]
	for i in range(len(membership)):
		cluster_id = membership[i]
		sib_clusters[cluster_id].append(labels[i])
	return sib_clusters

def create_X(vocabulary):
	"""
	Copied from hier_clust.py, because INPUT will change and
	don't want to mess with hier_clust.py too much.
	@inputs
	- vocabulary: set of concepts

	@outputs
	- X: list of lists, where each list represents a vector for a concept
	- labels: concepts, in the same order as its corresponding vector in X
	"""
	X = []
	labels = []

	f = open(INPUT, 'r')
	for line in f:
		word_vec = line.split()
		if word_vec[0] in vocabulary:
			X.append([float(x) for x in word_vec[1:]])
			labels.append(word_vec[0])

	return (X, labels)

def get_concept_domains():
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	X, labels = create_X(vocabulary)

	Z = hierarchy.linkage(X, method='average', metric='cosine')

	sib_clusters = distance_siblings(Z, labels, 40) # 0.87
	new_clusters = []
	new_clust = []
	for cluster in sib_clusters:
		if len(cluster) < 7:
			new_clust.extend(cluster)
		else:
			new_clusters.append(cluster)
	new_clusters.append(new_clust)
	print len(new_clusters)

	concept_domains = {c: [] for c in vocabulary}
	for i, clust in enumerate(new_clusters):
		for c in clust:
			concept_domains[c].append(i)

	return concept_domains

def get_concept_domains_lda():
	"""
	concept: domain string
	"""
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	d_file = open(DOMAINS, 'r')
	concept_domains = {}
	for line in d_file:
		contents = line.split()
		values = [float(x) for x in contents[1:]]
		if contents[0] in vocabulary:
			concept_domains[contents[0]] = [i for i in range(len(values))
			if values[i] == max(values)]
	return concept_domains


def get_concept_domains_old():
	'''
	concept: domain string
	'''
	vocab_file = open(VOCAB, 'r')
	vocabulary = set()
	for line in vocab_file:
		vocabulary.add(line.strip())

	# strange edge case of bluejay = blue jay = jaybird in wordnet
	if 'bluejay' in vocabulary:
		vocabulary.remove('bluejay')
		vocabulary.add('jaybird')

	offset_to_domain = {}
	domain_map = open(DOMAINS, 'r')
	for line in domain_map:
		contents = line.split()
		offset_to_domain[contents[0]] = contents[2:] # taking the first domain. each sense can have multiple domains

	concept_domains = {} # concept: [domains]
	for concept in vocabulary:
		senses = wn.synsets(concept)
		offset = str(senses[0].offset()).zfill(8) + '-' + senses[0].pos() # the first sense
		assert senses[0].pos() == 'n' # should at least be a noun
		if offset not in offset_to_domain: # there are some concepts without domain labels
			concept_domains[concept] = ['n/a']
		else:
			concept_domains[concept] = offset_to_domain[offset]
	return concept_domains

def get_domain_concepts():
	concept_domains = get_concept_domains()
	domain_concepts = {} # domain: [concepts]
	for concept in concept_domains:
		domains = concept_domains[concept]
		for d in domains:
			if d in domain_concepts:
				domain_concepts[d].append(concept)
			else:
				domain_concepts[d] = [concept]
	return domain_concepts

if __name__ == '__main__':
	domain_concepts = get_domain_concepts()

	feature_fit = ['cigarette', 'rope', 'jar', 'bottle', 'hose', 'ashtray', 'envelope', 'bag', 'cigar', 'seaweed', 'stick', 'vine', 'brick', 'shell', 'toy', 'sack', 'saddle', 'helmet', 'mirror', 'clam', 'chandelier', 'muzzle', 'sandals', 'beehive', 'mat', 'alligator', 'bathtub', 'crab', 'python', 'emerald', 'paintbrush', 'willow', 'pencil', 'shrimp', 'menu', 'buckle', 'comb', 'marble', 'rat', 'walnut', 'turtle', 'bucket', 'trailer', 'snail', 'tent', 'ant', 'cork', 'housefly', 'pearl', 'raccoon', 'bread', 'crocodile', 'iguana', 'rabbit', 'rattlesnake', 'salamander', 'toilet', 'cockroach', 'cow', 'crowbar', 'frog', 'veil', 'wand', 'whip', 'eel', 'raisin', 'crayon', 'pony', 'worm', 'belt', 'cage', 'candle', 'cat', 'drapes', 'gopher', 'mouse', 'pillow', 'tortoise', 'armour', 'beetle', 'guppy', 'cabin', 'dish', 'thermometer', 'donkey', 'faucet', 'cheese', 'anchor', 'basket', 'broom', 'cup', 'gorilla', 'mug', 'plate', 'pot', 'spider', 'umbrella', 'walrus', 'wrench', 'spatula', 'bear', 'garlic', 'sardine', 'tongs', 'pepper', 'toad', 'dog', 'hamster', 'skunk', 'doll', 'squid', 'bowl', 'octopus', 'banner', 'barn', 'beans', 'bridge', 'curtains', 'spoon', 'barrel', 'box', 'catfish', 'goldfish', 'lobster', 'saucer', 'tiger', 'porcupine', 'ruler', 'rake', 'coin', 'groundhog', 'balloon', 'flea', 'hare', 'colander', 'door', 'dresser', 'fence', 'grater', 'horse', 'hut', 'kite', 'lamb', 'lion', 'oak', 'otter', 'pier', 'pig', 'shack', 'shelves', 'sled', 'sledgehammer', 'sleigh', 'slingshot', 'tray', 'wall', 'napkin', 'biscuit', 'elephant', 'escalator', 'radio', 'rice', 'thimble', 'urn', 'zebra', 'beaver', 'fox', 'ladle', 'sink', 'brush', 'calf', 'squirrel', 'deer', 'mackerel', 'mink', 'moose', 'oven', 'perch', 'strainer', 'tuna', 'whale', 'ball', 'bench', 'blender', 'carpet', 'freezer', 'kettle', 'limousine', 'pen', 'penguin', 'salmon', 'surfboard', 'trout', 'vest', 'pan', 'skillet', 'chimp', 'cod', 'corn', 'cougar', 'fawn', 'goat', 'pie', 'seal', 'tractor', 'tricycle', 'fork', 'moth', 'cloak', 'corkscrew', 'dandelion', 'dishwasher', 'fridge', 'lantern', 'mixer', 'raft', 'stove', 'table', 'tap', 'toaster', 'baton', 'coconut', 'peg', 'rattle', 'spade', 'trolley', 'chipmunk', 'grasshopper', 'microwave', 'minnow', 'wasp', 'banana', 'cake', 'earmuffs', 'olive', 'parsley', 'pear', 'sandpaper', 'sheep', 'ambulance', 'buffalo', 'buzzard', 'chair', 'cheetah', 'couch', 'cushion', 'dolphin', 'giraffe', 'hyena', 'leopard', 'ostrich', 'panther', 'seagull', 'turkey', 'turnip', 'apron', 'hornet', 'shield', 'wagon', 'wheelbarrow', 'bayonet', 'boots', 'camel', 'coyote', 'elk', 'avocado', 'cape', 'caribou', 'caterpillar', 'duck', 'pliers', 'rooster', 'strawberry', 'peas', 'book', 'drain', 'football', 'ox', 'pickle', 'platypus', 'racquet', 'rocket', 'wheel', 'skis', 'grenade', 'stone', 'crown', 'goose', 'stereo', 'stork', 'desk', 'garage', 'rock', 'screwdriver', 'unicycle', 'mushroom', 'skateboard', 'bedroom', 'bull', 'certificate', 'peacock', 'potato', 'rhubarb', 'yam', 'radish', 'bracelet', 'microscope', 'spear', 'bison', 'chain', 'emu', 'cupboard', 'pyramid', 'bolts', 'knife', 'tack', 'lamp', 'raspberry', 'pin', 'screws', 'sword', 'celery', 'buggy', 'bullet', 'doorknob', 'gate', 'machete', 'medal', 'missile', 'razor', 'scissors', 'tripod', 'bra', 'onions', 'scarf', 'cucumber', 'magazine', 'zucchini', 'telephone', 'bike', 'blueberry', 'eagle', 'pumpkin', 'boat', 'jacket', 'revolver', 'clock', 'carrot', 'subway', 'grape', 'hammer', 'plum', 'sailboat', 'butterfly', 'chicken', 'helicopter', 'parka', 'pistol', 'eggplant', 'elevator', 'beets', 'hook', 'robe', 'shoes', 'slippers', 'socks', 'asparagus', 'bed', 'canoe', 'honeydew', 'inn', 'nightgown', 'rocker', 'sofa', 'apple', 'broccoli', 'cabbage', 'cauliflower', 'house', 'tomato', 'cranberry', 'gloves', 'grapefruit', 'lettuce', 'mandarin', 'mittens', 'spinach', 'cannon', 'bouquet', 'jeans', 'swan', 'coat', 'key', 'birch', 'building', 'cedar', 'dress', 'shed', 'skirt', 'trousers', 'whistle', 'church', 'dagger', 'hoe', 'pajamas', 'pants', 'vulture', 'cathedral', 'hatchet', 'sweater', 'scooter', 'taxi', 'chisel', 'nylons', 'peach', 'cantaloupe', 'crow', 'drill', 'level', 'lime', 'necklace', 'blouse', 'pine', 'shirt', 'woodpecker', 'airplane', 'bookcase', 'budgie', 'bureau', 'cabinet', 'closet', 'hawk', 'pelican', 'shawl', 'shovel', 'tie', 'camisole', 'canary', 'chickadee', 'dove', 'falcon', 'jet', 'oriole', 'owl', 'parakeet', 'raven', 'basement', 'cart', 'partridge', 'pheasant', 'pigeon', 'robin', 'starling', 'submarine', 'flute', 'finch', 'flamingo', 'swimsuit', 'gun', 'bluejay', 'nightingale', 'sparrow', 'blackbird', 'cherry', 'lemon', 'nectarine', 'pineapple', 'prune', 'cellar', 'leotards', 'tangerine', 'clamp', 'orange', 'skyscraper', 'piano', 'axe', 'bazooka', 'bomb', 'catapult', 'crossbow', 'harpoon', 'rifle', 'shotgun', 'tomahawk', 'bus', 'cottage', 'train', 'yacht', 'motorcycle', 'car', 'jeep', 'apartment', 'bungalow', 'van', 'ship', 'truck', 'gown', 'drum', 'typewriter', 'cello', 'tuba', 'harp', 'harpsichord', 'accordion', 'banjo', 'violin', 'clarinet', 'bagpipe', 'guitar', 'harmonica', 'saxophone', 'trombone', 'trumpet']

	for d in domain_concepts:
		print(d, domain_concepts[d])
		ranks = [feature_fit.index(c) for c in domain_concepts[d] if c in feature_fit]
		print("mean", np.mean(ranks), "med", np.median(ranks), "min", np.min(ranks), "max", np.max(ranks))
		print("\n")
