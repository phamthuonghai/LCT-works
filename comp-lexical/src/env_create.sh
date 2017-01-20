#!/bin/bash

virtualenv -p python --no-site-packages env
./env/bin/pip install tqdm nltk