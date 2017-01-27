#!/bin/bash

bash ./data_get.sh
bash ./data_preprocess.sh


./env/bin/python ./data_convert.py ./bilingual_data/en-vi.txt ./bilingual_data/en-vi.align ./bilingual_data/en-vi.rows ./bilingual_data/en-vi.cols ./bilingual_data/en-vi.sm
head -n5 ./bilingual_data/en-vi.rows > ./bilingual_data/list_top5_highest.txt
tail -n5 ./bilingual_data/en-vi.rows > ./bilingual_data/list_top5_lowest.txt
