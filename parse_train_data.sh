#!/bin/bash

if [ "$#" -eq "0" ]; then
	log_file="$(ls -1t logs/*.log | head -n1)"
else
	log_file="$1"
fi

step=$(mktemp)
egrep -o '.* Dev-matched cost: [0-9.]+' $log_file | egrep -o 'Step: [0-9]+' | egrep -o '[0-9]+' > $step

infe=$(mktemp)
egrep -o 'Inference value: Mean=[0-9.]+' $log_file | egrep -o '[0-9.]+' > $infe

contra=$(mktemp)
egrep -o 'Contradiction value: Mean=[0-9.]+' $log_file | egrep -o '[0-9.]+' > $contra

neutral=$(mktemp)
egrep -o 'Neutral value: Mean=[0-9.]+' $log_file | egrep -o '[0-9.]+' > $neutral

original=$(mktemp)
egrep -o 'Train loss: Mean=[0-9.]+' $log_file | egrep -o '[0-9.]+' > $original

regularized=$(mktemp)
egrep -o 'Regularized loss: Mean=[0-9.]+' $log_file | egrep -o '[0-9.]+' > $regularized

dev=$(mktemp)
egrep -o 'Dev-matched cost: [0-9.]+' $log_file | egrep -o '[0-9.]+' > $dev

echo "$log_file" && echo -e 'Step\tOriginalLoss\tDevOriginalLoss\tRegularizedLoss\tInference\tContradition\tNeutral' &&  paste $step $original $dev $regularized $infe $contra $neutral
