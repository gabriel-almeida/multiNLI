#!/bin/bash

KEEP_RATE="0.9"
MODEL_PATH="model_$(date +"%d-%m-%y-%H-%M")"
DATA_PATH="/home/gabriel/PycharmProjects/entailment/datasets"

source activate py27
cd python
python train_mnli.py --emb_train --datapath=$DATA_PATH --keep_rate=$KEEP_RATE bilstm $MODEL_PATH
