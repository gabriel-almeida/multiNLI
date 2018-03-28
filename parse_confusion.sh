#!/bin/bash
if [ "$#" -eq 0 ]; then
        log_file=$(ls -1t logs/*.log | head -n1)
else
        log_file="$1"
fi

step=$(mktemp)
egrep -o '.* Dev-matched cost: [0-9.]+' $log_file | egrep -o 'Step: [0-9]+' | egrep -o '[0-9]+' > $step

confusion00=$(mktemp)
egrep -o '\(0, 0\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion00

confusion01=$(mktemp)
egrep -o '\(0, 1\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion01

confusion02=$(mktemp)
egrep -o '\(0, 2\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion02

confusion10=$(mktemp)
egrep -o '\(1, 0\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion10

confusion11=$(mktemp)
egrep -o '\(1, 1\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion11

confusion12=$(mktemp)
egrep -o '\(1, 2\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion12

confusion20=$(mktemp)
egrep -o '\(2, 0\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion20

confusion21=$(mktemp)
egrep -o '\(2, 1\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion21

confusion22=$(mktemp)
egrep -o '\(2, 2\): [0-9]+' $log_file | egrep -o '[0-9]+$' > $confusion22

positives=$(mktemp)
paste -d+ $confusion00 $confusion11 $confusion22 | bc > $positives

result=$(echo "$log_file" && echo -e 'Step\tPositives\tTarget0Pred0\tTarget0Pred1\tTarget0Pred2\tTarget1Pred0\tTarget1Pred1\tTarget1Pred2\tTarget2Pred0\tTarget2Pred1\tTarget2Pred2' && paste $step $positives $confusion00 $confusion01 $confusion02 $confusion10 $confusion11 $confusion12 $confusion20 $confusion21 $confusion22)

echo "$result"
best=$(echo "$result" | sort -k2 -n -r | head -1)

echo "BEST:" && echo "$best"

#total=$(echo "$best" | cut -f 3-11 | tr $'\t' + | bc)

