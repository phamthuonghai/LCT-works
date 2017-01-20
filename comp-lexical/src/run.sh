#!/bin/bash
bash ./env_create.sh

bash ./data_get.sh
bash ./data_preprocess.sh


python ./data_convert.py
