#!/usr/bin/env bash

wget -r -P ./datasets ftp://ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/
gunzip ./datasets/ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/*.gz
java -jar ./java/mol2csv.jar ./datasets/ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/
python ./python/csvs2graphs.py ./datasets/ftp.ics.uci.edu/pub/baldig/learning/nci/gi50/ ./splits