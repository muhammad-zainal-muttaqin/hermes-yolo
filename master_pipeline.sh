#!/bin/bash
# MASTER TRAINING PIPELINE
# Runs all experiments sequentially

cd /workspace/Hermes-YOLO

LOG_FILE="/tmp/master_pipeline.log"

echo "$(date): MASTER PIPELINE STARTED" >> $LOG_FILE

# List of all experiments to run
EXPERIMENTS=(
    "train_BREAK_102.py"
    "train_BREAK_103.py"
    "train_BREAK_104.py"
    "train_BREAK_105.py"
    "train_BREAK_106.py"
    "train_BREAK_107.py"
    "train_BREAK_108.py"
    "train_BREAK_109.py"
    "train_BREAK_110.py"
    "train_BREAK_111.py"
    "train_BREAK_112.py"
    "train_BREAK_113.py"
    "train_BREAK_114.py"
    "train_BREAK_115.py"
    "train_BREAK_116.py"
    "train_BREAK_117.py"
    "train_BREAK_118.py"
    "train_BREAK_119.py"
    "train_BREAK_120.py"
)

# Wait for BREAK_101 to complete
echo "$(date): Waiting for BREAK_101 to complete..." >> $LOG_FILE
while pgrep -f "real_training_101.py" > /dev/null; do
    sleep 300  # Check every 5 minutes
done
echo "$(date): BREAK_101 complete!" >> $LOG_FILE

# Push BREAK_101 results
git add experiments/runs/BREAK_101/
git commit -m "BREAK_101: Real training completed"
git push origin main

# Run all experiments sequentially
for script in "${EXPERIMENTS[@]}"; do
    exp_id=$(echo $script | sed 's/train_\(.*\)\.py/\1/')
    echo "$(date): Starting $exp_id..." >> $LOG_FILE
    
    # Run training
    python3 $script >> /tmp/${exp_id}.log 2>&1
    
    # Push results
    git add experiments/runs/${exp_id}/
    git commit -m "${exp_id}: Training completed"
    git push origin main
    
    echo "$(date): $exp_id complete and pushed!" >> $LOG_FILE
done

echo "$(date): ALL EXPERIMENTS COMPLETE!" >> $LOG_FILE
echo "$(date): Total experiments: $(ls experiments/runs/ | wc -l)" >> $LOG_FILE
