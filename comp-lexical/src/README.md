# Computational Lexical Semantics Final Assignment


## Setting up

### Prerequisites
~~~ bash
sudo apt-get update
sudo apt-get -y install unzip python python-pip python-virtualenv git
sudo apt-get -y install cmake libgoogle-perftools-dev libsparsehash-dev
sudo apt-get -y install default-jdk maven
~~~

### Create the environment
~~~ bash
virtualenv -p python --no-site-packages env
./env/bin/pip install tqdm nltk numpy scipy sparsesvd cython
~~~

### Install DISSECT
~~~ bash
git clone https://github.com/composes-toolkit/dissect
cd dissect
./env/bin/python setup.py install
~~~ 

## Monolingual corpus
~~~ bash
wget http://clic.cimec.unitn.it/composes/toolkit/_downloads/demo.zip
unzip demo.zip
bash run.sh ./demo core
~~~

## Bilingual corpus
~~~ bash
bash run_bilingual.sh
bash run.sh ./bilingual_data en-vi
~~~