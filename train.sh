#!/bin/bash

KEEP_RATE="0.9"
MODEL_PATH="model_$(date +"%d-%m-%y-%H-%M")"
DATA_PATH="/home/gabrielgar_almeida/multiNLI/data"

cd python
LD_LIBRARY_PATH=/home/gabrielgar_almeida/cuda_cudnn5/lib64 python train_mnli.py --emb_train --datapath=$DATA_PATH --keep_rate=$KEEP_RATE bilstm $MODEL_PATH
