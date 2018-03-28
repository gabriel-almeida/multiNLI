#!/bin/bash

if [ "$#" -eq 0 ]; then
	log_file=$(ls -1t logs/*.log | head -n1)
else
	log_file="$1"
fi

watch tail -n40 $log_file
