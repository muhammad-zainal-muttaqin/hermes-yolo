#!/bin/bash
# Autonomous Research Runner - 8 Hours Max
# Runs experiments, commits results, and pushes to GitHub

set -e

REPO_DIR="/workspace/Hermes-YOLO"
LOG_FILE="$REPO_DIR/experiments/logs/autorun.log"
PID_FILE="/tmp/autoresearch.pid"
START_TIME=$(date +%s)
MAX_HOURS=8
MAX_SECONDS=$((MAX_HOURS * 3600))

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Save PID
echo $$ > "$PID_FILE"

# Initialize logging
mkdir -p "$REPO_DIR/experiments/logs"
log "🚀 Starting 8-Hour Autonomous Research Pipeline"
log "📁 Repository: $REPO_DIR"
log "⏱️  Max duration: $MAX_HOURS hours"
log "🎯 Target: Maximize mAP50 with novel approaches"

# Check dataset
check_dataset() {
    DATASET_DIR="/root/.cache/huggingface/hub"
    if [ -d "$DATASET_DIR" ]; then
        SIZE=$(du -sh "$DATASET_DIR" 2>/dev/null | cut -f1)
        info "Dataset cache: $SIZE"
        return 0
    fi
    return 1
}

# Wait for dataset (max 30 min)
wait_for_dataset() {
    info "Checking for dataset availability..."
    ATTEMPTS=0
    MAX_ATTEMPTS=30
    
    while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
        if check_dataset; then
            log "✅ Dataset ready!"
            return 0
        fi
        info "Waiting for dataset... attempt $((ATTEMPTS+1))/$MAX_ATTEMPTS"
        sleep 60
        ATTEMPTS=$((ATTEMPTS+1))
    done
    
    error "Dataset not ready after 30 min, proceeding anyway..."
    return 1
}

# Run single experiment
run_experiment() {
    local EXP_NUM=$1
    local EXP_ID=$(printf "AR_%03d" $EXP_NUM)
    
    log "🧪 Starting experiment $EXP_ID"
    
    # Create Python script for single experiment
    local PY_SCRIPT="$REPO_DIR/temp_exp_${EXP_NUM}.py"
    
    cat > "$PY_SCRIPT" << 'PYTHON_EOF'
import sys
import os
sys.path.insert(0, '/workspace/Hermes-YOLO')

from autoresearch import AutonomousResearch, NovelIdeaGenerator, ExperimentRunner, ResearchLedger, GitHubSync, Visualizer, SEED
import traceback

# Initialize
ledger = ResearchLedger()
idea_gen = NovelIdeaGenerator(seed=SEED)
runner = ExperimentRunner()
git = GitHubSync()
viz = Visualizer()

exp_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Generate and run single experiment
config = idea_gen.generate_experiment(exp_num)
exp_idx = ledger.add_experiment(config, "running")

try:
    # Run training
    results = runner.run_experiment(config)
    
    # Update ledger
    ledger.update_experiment(exp_idx, results)
    
    # Generate visualizations
    viz.generate_report(ledger.ledger)
    
    # Push to GitHub
    commit = git.push_experiment(
        config.id,
        f"{config.id}: {config.name} - mAP50={results['map50']:.4f}"
    )
    
    if commit:
        ledger.ledger[exp_idx]["git_commit"] = commit
        ledger.save_ledger()
    
    print(f"SUCCESS:{results['map50']:.4f}")
    
except Exception as e:
    error_msg = traceback.format_exc()
    ledger.ledger[exp_idx]["status"] = "failed"
    ledger.ledger[exp_idx]["notes"].append(f"Error: {str(e)}")
    ledger.save_ledger()
    print(f"FAILED:{str(e)}")
    sys.exit(1)
PYTHON_EOF

    # Run the experiment
    cd "$REPO_DIR"
    if timeout 3600 python "$PY_SCRIPT" "$EXP_NUM" 2>&1 | tee -a "$LOG_FILE"; then
        RESULT=$(tail -5 "$LOG_FILE" | grep "SUCCESS" | tail -1)
        log "✅ $EXP_ID completed: $RESULT"
        rm -f "$PY_SCRIPT"
        return 0
    else
        error "$EXP_ID failed or timed out"
        rm -f "$PY_SCRIPT"
        return 1
    fi
}

# Calculate time remaining
time_remaining() {
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    REMAINING=$((MAX_SECONDS - ELAPSED))
    echo $((REMAINING / 60))  # Return minutes
}

# Main loop
main() {
    # Wait for dataset
    wait_for_dataset
    
    EXP_NUM=0
    
    while true; do
        REMAINING_MIN=$(time_remaining)
        
        if [ $REMAINING_MIN -lt 30 ]; then
            log "⏰ Less than 30 min remaining. Wrapping up..."
            break
        fi
        
        log "⏱️  Time remaining: ${REMAINING_MIN} min | Running experiment $EXP_NUM"
        
        if run_experiment $EXP_NUM; then
            log "📤 Experiment $EXP_NUM pushed to GitHub"
        else
            error "Experiment $EXP_NUM failed, continuing..."
        fi
        
        EXP_NUM=$((EXP_NUM + 1))
        
        # Safety: max 50 experiments
        if [ $EXP_NUM -ge 50 ]; then
            log "🛑 Reached max experiment count (50)"
            break
        fi
        
        # Small delay between experiments
        sleep 10
    done
    
    # Final push
    log "🏁 Final push and summary generation..."
    cd "$REPO_DIR"
    
    # Generate final summary
    python -c "
import sys
sys.path.insert(0, '/workspace/Hermes-YOLO')
from autoresearch import ResearchLedger, Visualizer, GitHubSync

ledger = ResearchLedger()
viz = Visualizer()
git = GitHubSync()

viz.generate_report(ledger.ledger)
git.push_experiment('final', 'Final: 8-hour autonomous research complete')

best = ledger.get_best_experiment()
if best:
    print(f'Best: {best[\"config\"][\"id\"]} mAP50={best[\"results\"][\"map50\"]:.4f}')
"
    
    # Cleanup
    rm -f "$PID_FILE"
    
    log "✅ Autonomous research complete!"
    log "📊 Check GitHub for full results"
}

# Trap signals
trap 'error "Interrupted"; rm -f "$PID_FILE" "$REPO_DIR"/temp_exp_*.py; exit 1' INT TERM

# Run main
main
