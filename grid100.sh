#!/bin/bash

cd python

keep_rates=( "0.9" )
pis=( "0.25" )

for keep in "${keep_rates[@]}"
do
	for pi in "${pis[@]}"
	do
		echo "KEEP=${keep} PI=${pi}"
		python train_mnli.py --emb_train --datapath="/home/gabrielgar_almeida/multiNLI/data" \
		--keep_rate="${keep}" --epochs=10 --patience=10 --display_step_ratio=0.1 --pi="${pi}" --train_file="multinli_1.0_train.jsonl" bilstm "grid_bilstm_100_pi-${pi}_keep-${keep}_hybrid"
	done
#multinli_25.jsonl
#multinli_1.0_train.jsonl
done
echo "Finished - shutting down in 30 s" && sleep 30 && poweroff 
