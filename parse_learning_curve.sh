#!/bin/bash

if [ "$#" -eq 0 ]; then
	log_file=$(ls -1t logs/*.log | head -n1)
else
	log_file="$1"
fi

step=$(mktemp)
egrep -o '.* Dev-matched cost: [0-9.]+' $log_file | egrep -o 'Step: [0-9]+' | egrep -o '[0-9]+' > $step

infe=$(mktemp)
egrep -o 'Dev inference rule: .*= [0-9.]+' $log_file | egrep -o '[0-9.]+$' > $infe

contra=$(mktemp)
egrep -o 'Dev contradiction rule: .*= [0-9.]+' $log_file | egrep -o '[0-9.]+$' > $contra

neutral=$(mktemp)
egrep -o 'Dev neutral rule: .*= [0-9.]+' $log_file | egrep -o '[0-9.]+$' > $neutral

val=$(mktemp)
egrep -o 'Dev-matched acc: [0-9.]+' $log_file | egrep -o '[0-9.]+' | sed -e 's/$/*100/' | bc > $val

echo "$log_file" && echo -e '$Step$ $ValidationAcc$ $Inference$ $Contradition$ $Neutral$' && paste -d' ' $step $val $infe $contra $neutral
