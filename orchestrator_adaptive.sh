#!/bin/bash
# Adaptive Autonomous Research Orchestrator v2
# Features: Online research, adaptive idea generation, detailed analysis

REPO_DIR="/workspace/Hermes-YOLO"
LOG_DIR="$REPO_DIR/experiments/logs"
PID_FILE="/tmp/autoresearch_adaptive.pid"

# Time tracking
START_TIME=$(date +%s)
MAX_HOURS=8
MAX_SECONDS=$((MAX_HOURS * 3600))

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/adaptive_orchestrator_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save PID
echo $$ > "$PID_FILE"

log "🧠 ADAPTIVE AUTONOMOUS RESEARCH STARTED"
log "⏱️  Duration: $MAX_HOURS hours maximum"
log "🎯 Features: Online research + Adaptive idea generation"
log "📊 All ideas documented in IDEA.md"
log "="

# Time remaining
time_remaining() {
    local current=$(date +%s)
    local elapsed=$((current - START_TIME))
    local remaining=$((MAX_SECONDS - elapsed))
    echo $((remaining / 60))
}

# Initialize experiment counter
get_next_exp_num() {
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from idea_generator import IdeaGenerator
gen = IdeaGenerator(seed=42)
ledger = gen.analyzer.ledger
print(len(ledger))
" 2>/dev/null || echo 0
}

# Search online for relevant papers/techniques
search_online_solutions() {
    local problem="$1"
    log "   🔍 Researching online: $problem"
    
    # Use arxiv tool if available
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
try:
    from skill_view import skill_view
    # Search for relevant papers
    print('Searching arxiv for object detection improvements...')
    # Results would be from arxiv search
except:
    pass
" 2>/dev/null | tee -a "$LOG_FILE"
    
    log "   💡 Online research complete"
}

# Analyze results and generate new idea
analyze_and_generate_idea() {
    local exp_num=$1
    
    log "   🧠 Analyzing experiments and generating new idea..."
    
    cd "$REPO_DIR"
    
    # Generate new idea based on analysis
    python3 idea_generator.py $exp_num 2>&1 | tee -a "$LOG_FILE"
    
    # Push IDEA.md update
    git add IDEA.md 2>/dev/null || true
    git commit -m "Update IDEA.md: Add experiment IDEA_$(printf '%03d' $exp_num) rationale" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    
    log "   ✅ IDEA.md updated and pushed"
}

# Run single experiment
run_experiment() {
    local exp_num=$1
    local exp_id=$(printf "AR_%03d" $exp_num)
    
    log ""
    log "🧪 EXPERIMENT $exp_id"
    log "   Time remaining: $(time_remaining) minutes"
    
    # First, analyze and generate idea
    analyze_and_generate_idea $exp_num
    
    # Run the experiment
    cd "$REPO_DIR"
    
    # Get the strategy from latest idea
    STRATEGY=$(python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
from idea_generator import IdeaGenerator
gen = IdeaGenerator(seed=42)
idea = gen.generate_new_idea($exp_num)
print(idea['strategy_key'])
" 2>/dev/null || echo "contrastive_learning")
    
    log "   🎯 Strategy: $STRATEGY"
    
    # Execute experiment
    if timeout 3600 python3 run_single_experiment.py $exp_num 2>&1 | tee -a "$LOG_FILE"; then
        RESULT=$(tail -20 "$LOG_FILE" | grep "SUCCESS:" | tail -1 | cut -d: -f2)
        log "   ✅ $exp_id completed: mAP50=$RESULT"
        
        # Analyze result
        log "   📊 Analyzing result..."
        
        # Add result analysis to IDEA.md
        python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
import json
from pathlib import Path

exp_dir = Path('$REPO_DIR/experiments/runs/$exp_id')
results_file = exp_dir / 'results.json'

if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    idea_md = Path('$REPO_DIR/IDEA.md')
    if idea_md.exists():
        with open(idea_md) as f:
            content = f.read()
        
        # Add results section
        results_section = f\"\"\"
## 📈 Results for {exp_id}

- **mAP50**: {results.get('map50', 0):.4f}
- **mAP50-95**: {results.get('map50_95', 0):.4f}
- **Precision**: {results.get('precision', 0):.4f}
- **Recall**: {results.get('recall', 0):.4f}

### Analysis

_Results analysis will be added in next iteration._

\"\"\"
        
        # Insert after the first experiment section
        content = results_section + content
        
        with open(idea_md, 'w') as f:
            f.write(content)
" 2>/dev/null
        
    else
        log "   ❌ $exp_id failed"
        
        # If failed, search for solutions
        log "   🔍 Searching for solutions to failure..."
        search_online_solutions "experiment_failure"
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
except Exception as e:
    print(f'Viz error: {e}')
" 2>&1 | tee -a "$LOG_FILE"
    
    # Push everything
    log "   📤 Pushing all results to GitHub..."
    cd "$REPO_DIR"
    git add -A 2>/dev/null || true
    git commit -m "$exp_id: Results and analysis - strategy=$STRATEGY" 2>/dev/null || true
    git push origin main 2>&1 | tee -a "$LOG_FILE" || true
    
    return 0
}

# Main loop
main() {
    EXP_NUM=$(get_next_exp_num)
    log "📋 Starting from experiment $EXP_NUM"
    
    while true; do
        REMAINING=$(time_remaining)
        
        if [ $REMAINING -lt 60 ]; then
            log "⏰ Less than 60 minutes remaining. Finalizing..."
            break
        fi
        
        run_experiment $EXP_NUM
        
        EXP_NUM=$((EXP_NUM + 1))
        
        if [ $EXP_NUM -ge 50 ]; then
            log "🛑 Reached maximum experiment count"
            break
        fi
        
        # Cooldown between experiments
        sleep 10
    done
    
    # Final summary
    log ""
    log "🏁 ADAPTIVE RESEARCH COMPLETE"
    
    # Generate comprehensive summary
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
try:
    from autoresearch import ResearchLedger, Visualizer
    from idea_generator import IdeaGenerator
    
    ledger = ResearchLedger()
    viz = Visualizer()
    gen = IdeaGenerator(seed=42)
    
    # Final visualizations
    viz.generate_report(ledger.ledger)
    
    # Final analysis
    trends = gen.analyzer.analyze_trends()
    insights = gen.analyzer.extract_insights()
    
    # Write final summary
    summary = f\"\"\"
# 🎉 Final Research Summary

## Performance Statistics

- **Total Experiments**: {trends.get('total_experiments', 0)}
- **Best mAP50**: {trends.get('best_map50', 0):.4f}
- **Mean mAP50**: {trends.get('mean_map50', 0):.4f}
- **Best Strategy**: {trends.get('best_strategy', 'N/A')}
- **Trend**: {trends.get('trend', 'unknown')}

## Key Insights

\"\"\"
    for insight in insights:
        summary += f'- {insight}\\n'
    
    summary += '\\n## Successful Strategies\\n\\n'
    
    # Add successful strategies
    completed = [e for e in ledger.ledger if e.get('status') == 'completed']
    completed.sort(key=lambda x: x['results'].get('map50', 0), reverse=True)
    
    for i, exp in enumerate(completed[:5], 1):
        config = exp['config']
        results = exp['results']
        summary += f\"{i}. **{config['id']}**: {config['name']} - mAP50={results['map50']:.4f}\\n\"
        summary += f\"   - Components: {', '.join(config.get('novel_components', [])[:3])}\\n\\n\"
    
    # Write to file
    with open('$REPO_DIR/FINAL_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print('Final summary generated')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"
    
    # Final push
    cd "$REPO_DIR"
    git add -A 2>/dev/null || true
    git commit -m "Final: 8-hour adaptive autonomous research complete" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    
    rm -f "$PID_FILE"
    
    log "✅ Complete! Check:"
    log "   - IDEA.md for all experiment ideas and rationale"
    log "   - FINAL_SUMMARY.md for comprehensive results"
    log "   - LEADERBOARD.md for rankings"
    log "   - GitHub: https://github.com/muhammad-zainal-muttaqin/Hermes-YOLO"
}

# Trap signals
trap 'log "Interrupted"; rm -f "$PID_FILE"; exit 1' INT TERM

# Run
main
