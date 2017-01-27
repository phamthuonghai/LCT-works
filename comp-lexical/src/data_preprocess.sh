#!/bin/bash
export SPARK_LINK="http://d3kbcqa49mib13.cloudfront.net/SPARK_FILE_NAME.tgz"
export SPARK_FILE_NAME="SPARK_FILE_NAME"
export SPARK_DIR="./spark"
export EN_POS_FILE="./bilingual_data/en.pos.txt"
export VI_POS_FILE="./bilingual_data/vi.pos.txt"
export EN_VI_FILE="./bilingual_data/en-vi.txt"
export ENV_PYTHON="./env/bin/python"

### Download and install Apache Spark
wget $SPARK_LINK
tar -xzf SPARK_FILE_NAME.tgz
mv SPARK_FILE_NAME spark
rm SPARK_FILE_NAME.tgz
"$SPARK_DIR/sbin/start-all.sh"

### Download and install vn.vitk https://github.com/phuonglh/vn.vitk
git clone git@github.com:phuonglh/vn.vitk.git
cd vn.vitk/
mvn compile package
sudo mkdir /export
sudo cp -r ./dat/ /export/
cd ..

### Use [vnTokenizer] to tokenize vi.txt to vi.token.txt
echo "======= Tokenizing VN text ======="
$SPARK_DIR/bin/spark-submit ./vn.vitk/target/vn.vitk-3.0.jar -i ./bilingual_data/OpenSubtitles2016.en-vi.vi -o ./bilingual_data/vi.token
cat ./bilingual_data/vi.token/part-* > ./bilingual_data/vi.token.txt
rm -rd ./bilingual_data/vi.token

### Use [vnTagger] to do POS tagging
echo "======= PoS tagging VN text ======="
$SPARK_DIR/bin/spark-submit --driver-memory 12g ./vn.vitk/target/vn.vitk-3.0.jar -t tag -a tag -i ./bilingual_data/vi.token.txt -o ./bilingual_data/vi.pos
cat ./bilingual_data/vi.pos/part-* > $VI_POS_FILE
rm -rd ./bilingual_data/vi.pos

echo "======= Tokenizing & PoS tagging EN text ======="
wget http://nlp.stanford.edu/software/stanford-postagger-2016-10-31.zip
unzip stanford-postagger-2016-10-31.zip
rm stanford-postagger-2016-10-31.zip
mv stanford-postagger-2016-10-31/ stanford-postagger
$ENV_PYTHON -u data_en_preprocess.py ./bilingual_data/OpenSubtitles2016.en-vi.en ./bilingual_data/en.token.txt $EN_POS_FILE

echo "======= Merging EN & VN text ======="
$ENV_PYTHON ./data_convert_pre_align.py $EN_POS_FILE $VI_POS_FILE $EN_VI_FILE


# Run [fast_align](https://github.com/clab/fast_align) for word alignment
echo "======= Running fast_align for word alignment ======="
git clone git@github.com:clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
./fast_align -i ../.$EN_VI_FILE -d -o -v -I 100 > ../../bilingual_data/en-vi.align
cd ../../
