mkdir -p mcrae
mkdir -p glove
mkdir -p word2vec
mkdir -p cslb

# McRae
wget https://static-content.springer.com/esm/art%3A10.3758%2FBF03192726/MediaObjects/McRae-BRM-2005.zip
unzip McRae-BRM-2005.zip
mv McRae-BRM-InPress/* mcrae/
rm McRae-BRM-2005.zip
rm -rf McRae-BRM-InPress/ 

# GloVE
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P glove
wget http://nlp.stanford.edu/data/glove.6B.zip -P glove
cd glove
unzip glove.840B.300d.zip && rm glove.840B.300d.zip
unzip glove.6B.zip && rm glove.6B.zip

# word2vec
cd ../word2vec 
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
cd ../

# CSLB 
echo "To download CSLB property norms, fill out a form at http://csl.psychol.cam.ac.uk/propertynorms/"
