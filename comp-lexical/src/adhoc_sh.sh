./env/bin/python data_en_preprocess.py ./bilingual_data/OpenSubtitles2016.en-vi.en ./bilingual_data/en.token.txt ./bilingual_data/en.pos.txt
./env/bin/python ./data_convert_pre_align.py ./bilingual_data/en.pos.txt ./bilingual_data/vi.pos.txt ./bilingual_data/en-vi.txt
cd ./fast_align/build/
./fast_align -i ../../bilingual_data/en-vi.txt -d -o -v -I 100 > ../../bilingual_data/en-vi.align
cd ../../
./env/bin/python ./data_convert.py ./bilingual_data/en-vi.txt ./bilingual_data/en-vi.align ./bilingual_data/en-vi.rows ./bilingual_data/en-vi.cols ./bilingual_data/en-vi.sm
head -n5 ./bilingual_data/en-vi.rows > ./bilingual_data/list_top5_highest.txt
tail -n5 ./bilingual_data/en-vi.rows > ./bilingual_data/list_top5_lowest.txt
