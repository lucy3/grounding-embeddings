"""
Get domains
"""

from nltk.corpus import wordnet as wn

VOCAB = "./all/vocab.txt"
DOMAINS = '../wndomains/wordnet-domains-3.2-wordnet-3.0.txt'

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

	offset_to_domain = {}
	domain_map = open(DOMAINS, 'r')
	for line in domain_map:
		contents = line.split()
		offset_to_domain[contents[0]] = contents[2:] # taking the first domain. each sense can have multiple domains

	concept_domains = {} # concept: [domains]
	for concept in vocabulary:
		if concept == 'dunebuggy':
			concept_domains[concept] = ['n/a']
			continue
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
	for domain in domain_concepts:
		print(domain, len(domain_concepts[domain]), domain_concepts[domain])
