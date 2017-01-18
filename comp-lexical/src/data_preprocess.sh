#!/bin/bash

### Use [vnTokenizer] to tokenize vi.txt to vi.token.txt
wget http://mim.hus.vnu.edu.vn/phuonglh/tools/vn.hus.nlp.tokenizer-4.1.1-bin.tar.gz
mkdir vn.hus.nlp.tokenizer-4.1.1-bin
tar -xvzf vn.hus.nlp.tokenizer-4.1.1-bin.tar.gz -C ./vn.hus.nlp.tokenizer-4.1.1-bin
cd ./vn.hus.nlp.tokenizer-4.1.1-bin
bash vnTokenizer.sh -i ../xces/vi.txt -o ../xces/vi.token.txt

### Use [vnTagger] to do POS tagging
wget http://mim.hus.vnu.edu.vn/phuonglh/tools/vn.hus.nlp.tagger-4.2.0-bin.tar.gz
mkdir vn.hus.nlp.tagger-4.2.0
tar -xvzf vn.hus.nlp.tagger-4.2.0-bin.tar.gz -C ./vn.hus.nlp.tagger-4.2.0
cd ./vn.hus.nlp.tagger-4.2.0


python ./data_convert_pre_align.py


# Run [fast_align](https://github.com/clab/fast_align) for word alignment
git clone git@github.com:clab/fast_align.git
sudo apt-get install libgoogle-perftools-dev libsparsehash-dev
cd fast_align
mkdir build
cd build
cmake ..
make
./fast_align -i ../../xces/en-vi.txt -d -o -v > ../../xces/en-vi.align
