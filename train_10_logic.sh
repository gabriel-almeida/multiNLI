#!/bin/bash

cd python
python train_mnli.py --emb_train --datapath="/home/gabrielgar_almeida/multiNLI/data" \
--keep_rate="0.9" --epochs=20 --pi=0.01 --train-file="multinli_10.jsonl" bilstm "10_bilstm_logic"
