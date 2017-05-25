hostname := $(shell hostname | grep -o "^[^.]\+")
datadir := /$(hostname)/scr1/jgauthie
glove := ~/scr/others/glove

norms := ../cslb/norms.dat

GCC := g++-4.8 -std=gnu++11

.PHONY: vocab.keep.txt cooccurrence.bin cooccurrence.filtered.bin cooccur.npz clean all
vocab.keep.txt: $(datadir)/vocab.keep.txt
cooccurrence.bin: $(datadir)/cooccurrence.bin
cooccurrence.filtered.bin: $(datadir)/cooccurrence.filtered.bin
cooccur.npz: $(datadir)/cooccur.npz


$(datadir)/vocab.keep.txt:
	python main.py --mode write-vocab --norms-file $(norms) --vocab-file $(datadir)/vocab.txt \
		--filtered-vocab-file $(datadir)/vocab.keep.txt

$(datadir)/cooccurrence.bin: $(datadir)/vocab.txt
	$(glove)/cooccur -memory 80 -vocab-file $(datadir)/vocab.txt -window-size 10 < $(datadir)/corpus > $(datadir)/cooccurrence.bin

$(datadir)/vocab.txt:
	$(glove)/vocab_count -min-count 5 < $(datadir)/corpus > $(datadir)/vocab.txt

$(datadir)/cooccurrence.filtered.bin: filter_glove
	./filter_glove $(datadir)

filter_glove:
	$(GCC) filter_glove.cc -o filter_glove

$(datadir)/cooccur.npz: $(datadir)/cooccurrence.filtered.bin
	python convert_to_npy.py --vocab_file $(datadir)/vocab.txt \
		--cooccur_file $(datadir)/cooccurrence.filtered.bin --out_file $(datadir)/cooccur.npz

ppmi_out.txt: $(datadir)/vocab.keep.txt $(datadir)/cooccur.npz
	python main.py --mode ppmi --norms-file $(norms) --vocab-file $(datadir)/vocab.txt \
		--filtered-vocab-file $(datadir)/vocab.keep.txt --cooccur-file $(datadir)/cooccur.npz \
		--cooccur-ppmi-file $(datadir)/cooccur.ppmi.npz > ppmi_out.txt

###############

clean:
	rm -f filter_glove $(datadir)/cooccurrence.filtered.bin $(datadir)/cooccur.npz \
		$(datadir)/cooccur.ppmi.npz $(datadir)/vocab.keep.txt

all: vocab.keep.txt cooccurrence.filtered.bin cooccur.npz ppmi_out.txt
