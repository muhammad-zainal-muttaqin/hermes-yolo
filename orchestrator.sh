#!/bin/bash
# Main Autonomous Research Orchestrator
# Runs for 8 hours, executing experiments iteratively

REPO_DIR="/workspace/Hermes-YOLO"
LOG_DIR="$REPO_DIR/experiments/logs"
PID_FILE="/tmp/autoresearch_main.pid"

# Time tracking
START_TIME=$(date +%s)
MAX_HOURS=8
MAX_SECONDS=$((MAX_HOURS * 3600))

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save PID
echo $$ > "$PID_FILE"

log "🚀 AUTONOMOUS RESEARCH STARTED"
log "⏱️  Duration: $MAX_HOURS hours maximum"
log "🎯 Target: Maximize mAP50 with novel approaches"
log "📊 All results will be pushed to GitHub"
log "="

# Time remaining function
time_remaining() {
    local current=$(date +%s)
    local elapsed=$((current - START_TIME))
    local remaining=$((MAX_SECONDS - elapsed))
    echo $((remaining / 60))  # minutes
}

# Initialize experiment counter from ledger
get_next_exp_num() {
    if [ -f "$REPO_DIR/experiments/logs/research_ledger.json" ]; then
        python3 -c "
import json
try:
    with open('$REPO_DIR/experiments/logs/research_ledger.json') as f:
        ledger = json.load(f)
    print(len(ledger))
except:
    print(0)
"
    else
        echo 0
    fi
}

EXP_NUM=$(get_next_exp_num)
log "📋 Starting from experiment $EXP_NUM"

# Main loop
while true; do
    REMAINING=$(time_remaining)
    
    # Check time
    if [ $REMAINING -lt 45 ]; then
        log "⏰ Less than 45 minutes remaining. Stopping to ensure clean finish."
        break
    fi
    
    EXP_ID=$(printf "AR_%03d" $EXP_NUM)
    log ""
    log "🧪 EXPERIMENT $EXP_ID"
    log "   Time remaining: ${REMAINING} minutes"
    
    # Run experiment with timeout (max 1 hour per experiment)
    cd "$REPO_DIR"
    
    if timeout 3600 python3 run_single_experiment.py $EXP_NUM 2>&1 | tee -a "$LOG_FILE"; then
        log "   ✅ $EXP_ID completed successfully"
        
        # Extract result
        MAP50=$(tail -20 "$LOG_FILE" | grep "SUCCESS:" | tail -1 | cut -d: -f2)
        log "   📊 mAP50: $MAP50"
    else
        log "   ❌ $EXP_ID failed or timed out"
    fi
    
    # Update visualizations
    log "   📈 Updating visualizations..."
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
try:
    from autoresearch import ResearchLedger, Visualizer
    ledger = ResearchLedger()
    viz = Visualizer()
    viz.generate_report(ledger.ledger)
    print('Visualizations updated')
except Exception as e:
    print(f'Viz error: {e}')
" 2>&1 | tee -a "$LOG_FILE"
    
    # Push to GitHub
    log "   📤 Pushing to GitHub..."
    cd "$REPO_DIR"
    git add -A 2>/dev/null || true
    git commit -m "$EXP_ID: Autonomous research iteration" 2>/dev/null || true
    git push origin main 2>&1 | tee -a "$LOG_FILE" || true
    
    EXP_NUM=$((EXP_NUM + 1))
    
    # Max 50 experiments safety
    if [ $EXP_NUM -ge 50 ]; then
        log "🛑 Reached maximum experiment count (50)"
        break
    fi
    
    # Small cooldown
    sleep 5
done

# Final summary
log ""
log "🏁 AUTONOMOUS RESEARCH COMPLETE"
log "Generating final report..."

python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
try:
    from autoresearch import ResearchLedger, Visualizer, GitHubSync
    ledger = ResearchLedger()
    viz = Visualizer()
    git = GitHubSync()
    
    viz.generate_report(ledger.ledger)
    
    best = ledger.get_best_experiment()
    if best:
        print(f'🎉 Best: {best[\"config\"][\"id\"]} mAP50={best[\"results\"][\"map50\"]:.4f}')
    
    # Final push
    git.push_experiment('final', '8-hour autonomous research complete')
    print('Final push complete')
except Exception as e:
    print(f'Error: {e}')
"

# Cleanup
rm -f "$PID_FILE"

log "✅ All done! Check GitHub for full results."
log "📊 Repository: https://github.com/muhammad-zainal-muttaqin/Hermes-YOLO"
