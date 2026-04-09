#!/bin/bash
while true; do
    echo "$(date): $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
    if [ -f experiments/runs/BREAK_101/train/results.csv ]; then
        tail -1 experiments/runs/BREAK_101/train/results.csv
    fi
    sleep 60
done
