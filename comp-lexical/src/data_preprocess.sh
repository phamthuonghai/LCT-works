#!/bin/bash


### Download and install Apache Spark
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.6.tgz
tar -xzf spark-1.6.3-bin-hadoop2.6.tgz
mv spark-1.6.3-bin-hadoop2.6 spark
rm spark-1.6.3-bin-hadoop2.6.tgz
./spark/sbin/start-all.sh

### Download and install vn.vitk https://github.com/phuonglh/vn.vitk
git clone git@github.com:phuonglh/vn.vitk.git
cd vn.vitk/
mvn compile package
sudo mkdir /export
sudo cp -r ./dat/ /export/
cd ..

### Use [vnTokenizer] to tokenize vi.txt to vi.token.txt
echo "======= Tokenizing VN text ======="
./spark/bin/spark-submit ./vn.vitk/target/vn.vitk-3.0.jar -i ./bilingual_data/OpenSubtitles2016.en-vi.vi -o ./bilingual_data/vi.token
cat ./bilingual_data/vi.token/part-* > ./bilingual_data/vi.token.txt
rm -rd ./bilingual_data/vi.token

### Use [vnTagger] to do POS tagging
echo "======= PoS tagging VN text ======="
./spark/bin/spark-submit --driver-memory 12g ./vn.vitk/target/vn.vitk-3.0.jar -t tag -a tag -i ./bilingual_data/vi.token.txt -o ./bilingual_data/vi.pos
cat ./bilingual_data/vi.pos/part-* > ./bilingual_data/vi.pos.txt
rm -rd ./bilingual_data/vi.pos

echo "======= Tokenizing & PoS taggin EN text ======="
wget http://nlp.stanford.edu/software/stanford-postagger-2016-10-31.zip
unzip stanford-postagger-2016-10-31.zip
rm stanford-postagger-2016-10-31.zip
mv stanford-postagger-2016-10-31/ stanford-postagger
./env/bin/python -u data_en_preprocess.py ./bilingual_data/OpenSubtitles2016.en-vi.en ./bilingual_data/en.token.txt ./bilingual_data/en.pos.txt

echo "======= Merging EN & VN text ======="
./env/bin/python ./data_convert_pre_align.py


# Run [fast_align](https://github.com/clab/fast_align) for word alignment
echo "======= Running fast_align for word alignment ======="
git clone git@github.com:clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
./fast_align -i ../../bilingual_data/en-vi.txt -d -o -v -I 100 > ../../bilingual_data/en-vi.align
cd ../../
