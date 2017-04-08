"""
Get domains
"""

from nltk.corpus import wordnet as wn

VOCAB = "./all/vocab.txt"

def get_concept_domains():
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

	depth = 6
	concept_domains = {} # concept: domain
	for concept in vocabulary:
		senses = wn.synsets(concept)
		hypernym_path = senses[0].hypernym_paths()[0] # the first hypernym path of the first sense
		assert hypernym_path[0].name() == 'entity.n.01' # the main sense should be a noun. if this fails...
		if len(hypernym_path) > depth:
			concept_domains[concept] = hypernym_path[depth].name().split('.')[0]
		else:
			concept_domains[concept] = hypernym_path[-1].name().split('.')[0]
	return concept_domains

def get_domain_concepts():
	concept_domains = get_concept_domains()
	domain_concepts = {} # domain: [concepts]
	for concept in concept_domains:
		domain = concept_domains[concept]
		if domain in domain_concepts:
			domain_concepts[domain].append(concept)
		else:
			domain_concepts[domain] = [concept]
	return domain_concepts

if __name__ == '__main__':
	domain_concepts = get_domain_concepts()
	for domain in domain_concepts:
		print domain, len(domain_concepts[domain]), domain_concepts[domain]
