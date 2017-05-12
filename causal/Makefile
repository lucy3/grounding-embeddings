hostname := $(shell hostname | grep -o "^[^.]\+")
datadir := /$(hostname)/scr1/jgauthie

norms := ../cslb/norms.dat

GCC := g++-4.8 -std=gnu++11

.PHONY: vocab.keep.txt cooccurrence.filtered.bin cooccur.npz clean all
vocab.keep.txt: $(datadir)/vocab.keep.txt
cooccurrence.filtered.bin: $(datadir)/cooccurrence.filtered.bin
cooccur.npz: $(datadir)/cooccur.npz


$(datadir)/vocab.keep.txt:
	python main.py --mode write-vocab --norms-file $(norms) --vocab-file $(datadir)/vocab.txt \
		--filtered-vocab-file $(datadir)/vocab.keep.txt

$(datadir)/cooccurrence.filtered.bin: filter_glove
	./filter_glove $(datadir)

filter_glove:
	$(GCC) filter_glove.cc -o filter_glove

$(datadir)/cooccur.npz: cooccurrence.filtered.bin
	python convert_to_npy.py --vocab_file $(datadir)/vocab.txt \
		--cooccur_file $(datadir)/cooccurrence.filtered.bin --out_file $(datadir)/cooccur.npz

###############

clean:
	rm filter_glove $(datadir)/cooccurrence.filtered.bin $(datadir)/cooccur.npz \
		$(datadir)/vocab.keep.txt

all: vocab.keep.txt cooccurence.filtered.bin cooccur.npz
