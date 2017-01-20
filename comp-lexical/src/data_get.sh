#!/bin/bash

echo "======= Getting OpenSubtitles2016 data ======="
wget http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016%2Fen-vi.txt.zip
mkdir ./bilingual_data
unzip download.php\?f\=OpenSubtitles2016%2Fen-vi.txt.zip -d ./bilingual_data
rm download.php\?f\=OpenSubtitles2016%2Fen-vi.txt.zip
