#!/bin/bash
# Auto-push script for Tier 1 experiments

cd /workspace/Hermes-YOLO

# Check each experiment and push if complete
for i in 001 002 003 004; do
    EXP="BREAK_$i"
    if [ -f "experiments/runs/$EXP/results.json" ]; then
        # Check if already pushed
        if ! git log --oneline -20 | grep -q "$EXP"; then
            echo "🚀 Pushing $EXP..."
            git add experiments/runs/$EXP/LEADERBOARD.md 2>/dev/null || true
            git add experiments/runs/$EXP/results.json 2>/dev/null || true
            git add experiments/runs/$EXP/config.yaml 2>/dev/null || true
            git commit -m "Tier1: $EXP breakthrough experiment complete" || true
            git push origin main || true
        fi
    fi
done

echo "✅ Auto-push check complete"
