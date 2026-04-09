#!/bin/bash
cd /workspace/Hermes-YOLO

# Wait for BREAK_101 to complete
while pgrep -f "real_training_101.py" > /dev/null; do
    echo "$(date): BREAK_101 still running..."
    sleep 300  # Check every 5 minutes
done

echo "$(date): BREAK_101 complete! Starting BREAK_102..."
nohup python3 train_BREAK_102.py > /tmp/train_102.log 2>&1 &

# Wait for 102
while pgrep -f "train_BREAK_102.py" > /dev/null; do
    sleep 300
done

echo "$(date): BREAK_102 complete! Starting BREAK_103..."
nohup python3 train_BREAK_103.py > /tmp/train_103.log 2>&1 &

# Wait for 103
while pgrep -f "train_BREAK_103.py" > /dev/null; do
    sleep 300
done

echo "$(date): BREAK_103 complete! Starting BREAK_104..."
nohup python3 train_BREAK_104.py > /tmp/train_104.log 2>&1 &

echo "$(date): All sequential training started!"
