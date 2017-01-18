#!/bin/bash

echo "Getting OpenSubtitles2016 data"
wget -r --no-parent --reject "index.html*" http://opus.lingfil.uu.se/OpenSubtitles2016/xml/en/ &
wget -r --no-parent --reject "index.html*" http://opus.lingfil.uu.se/OpenSubtitles2016/xml/vi/ &

wait

### Extract all gzip files
for file in ./*/*/*/*.gz; do gzip -dk $file; done
