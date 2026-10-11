#!/bin/bash

mkdir -p info

echo "Processando Amazon Fashion..."
# Note o ponto em ../.data/
python3 pro_data.py ../.data/AMAZON_FASHION_5.json amazon language_models/GoogleNews-vectors-negative300.bin word2vec > info/AMAZON_FASHION_w2v

echo "Processando All Beauty..."
# Note o ponto em ../.data/
python3 pro_data.py ../.data/All_Beauty.json amazon language_models/GoogleNews-vectors-negative300.bin word2vec > info/ALL_BEAUTY_w2v

echo "Concluído!"