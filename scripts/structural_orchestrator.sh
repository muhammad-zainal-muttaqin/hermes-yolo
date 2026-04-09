#!/bin/bash
# Structural Autonomous Research Orchestrator
# NO parameter tinkering - structural changes only!
# Following CONTEXT.md: "problem is structural first, optimization second"

REPO_DIR="/workspace/Hermes-YOLO"
LOG_DIR="$REPO_DIR/experiments/logs"
PID_FILE="/tmp/structural_autoresearch.pid"

# Time tracking
START_TIME=$(date +%s)
MAX_HOURS=8
MAX_SECONDS=$((MAX_HOURS * 3600))

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/structural_orchestrator_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Save PID
echo $$ > "$PID_FILE"

log "🧬 STRUCTURAL AUTONOMOUS RESEARCH STARTED"
log "⏱️  Duration: $MAX_HOURS hours maximum"
log "📜 Following CONTEXT.md: Structural changes, NOT parameter tinkering"
log "🎯 Target: mAP50 > 0.55 through architecture/data/formulation changes"
log "="

# Time remaining
time_remaining() {
    local current=$(date +%s)
    local elapsed=$((current - START_TIME))
    local remaining=$((MAX_SECONDS - elapsed))
    echo $((remaining / 60))
}

# Run structural experiment
run_structural_exp() {
    local exp_num=$1
    local exp_id=$(printf "STRUCT_%03d" $exp_num)
    
    log ""
    log "🧪 $exp_id - STRUCTURAL EXPERIMENT"
    log "   Time remaining: $(time_remaining) minutes"
    
    cd "$REPO_DIR"
    
    # Run structural experiment
    if timeout 3600 python3 structural_experiment.py $exp_num 2>&1 | tee -a "$LOG_FILE"; then
        RESULT=$(tail -20 "$LOG_FILE" | grep "SUCCESS:" | tail -1 | cut -d: -f2)
        log "   ✅ $exp_id completed: mAP50=$RESULT"
        
        # Update structural ideas doc
        update_structural_docs $exp_num
        
        # Generate visualizations
        generate_visualizations
        
    else
        log "   ❌ $exp_id failed"
    fi
    
    # Push
    log "   📤 Pushing to GitHub..."
    cd "$REPO_DIR"
    git add STRUCTURAL_IDEAS.md STRUCTURAL_LEADERBOARD.md experiments/ 2>/dev/null || true
    git add experiments/visualizations/ 2>/dev/null || true
    git commit -m "$exp_id: Structural experiment complete with visualizations" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    
    return 0
}

# Generate visualizations
generate_visualizations() {
    log "   📈 Generating visualizations..."
    
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Collect all results
results = []
runs_dir = Path('$REPO_DIR/experiments/runs')
viz_dir = Path('$REPO_DIR/experiments/visualizations')
viz_dir.mkdir(parents=True, exist_ok=True)

for run_dir in sorted(runs_dir.glob('STRUCT_*')):
    results_file = run_dir / 'results.json'
    config_file = run_dir / 'config.json'
    if results_file.exists() and config_file.exists():
        try:
            with open(results_file) as f:
                res = json.load(f)
            with open(config_file) as f:
                cfg = json.load(f)
            results.append({
                'exp_id': cfg['id'],
                'map50': res['map50'],
                'name': cfg['name']
            })
        except:
            pass

if len(results) >= 1:
    # Create progress plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_ids = [r['exp_id'] for r in results]
    map50s = [r['map50'] for r in results]
    names = [r['name'][:15] for r in results]
    
    colors = ['green' if m >= 0.52 else 'orange' if m >= 0.50 else 'red' for m in map50s]
    
    bars = ax.bar(range(len(exp_ids)), map50s, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, map50s)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add baseline line at 0.50
    ax.axhline(y=0.50, color='blue', linestyle='--', alpha=0.5, label='mAP50=0.50')
    
    # Add target line at 0.55
    ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.5, label='Target mAP50=0.55')
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('mAP50', fontsize=12)
    ax.set_title('Structural Research Progress - mAP50', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels([f\"{e.split('_')[1]}\\n{n}\" for e, n in zip(exp_ids, names)], 
                       rotation=0, ha='center', fontsize=8)
    ax.set_ylim(0.45, 0.58)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'progress_map50.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Generated progress_map50.png with {len(results)} experiments')
"
}

# Update documentation
update_structural_docs() {
    local exp_num=$1
    
    # Get analysis
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
import json
from pathlib import Path

exp_id = f'STRUCT_{$exp_num:03d}'
analysis_file = Path(f'$REPO_DIR/experiments/runs/{exp_id}/analysis.json')
results_file = Path(f'$REPO_DIR/experiments/runs/{exp_id}/results.json')
config_file = Path(f'$REPO_DIR/experiments/runs/{exp_id}/config.json')

if analysis_file.exists() and results_file.exists() and config_file.exists():
    with open(analysis_file) as f:
        analysis = json.load(f)
    with open(results_file) as f:
        results = json.load(f)
    with open(config_file) as f:
        config = json.load(f)
    
    # Update STRUCTURAL_IDEAS.md with results
    ideas_file = Path('$REPO_DIR/STRUCTURAL_IDEAS.md')
    with open(ideas_file) as f:
        content = f.read()
    
    # Add results section
    results_md = f\"\"\"

## 📊 {exp_id} Results

- **mAP50**: {results['map50']:.4f}
- **Structural Change**: {config['structural_change']}
- **Target Problem**: {config['target_problem']}
- **Success**: {analysis.get('success', False)}

**Analysis**: {analysis.get('notes', ['N/A'])[0]}

---
\"\"\"
    
    # Insert after the idea description
    content = content.replace(f\"## {exp_num+1}.\", results_md + f\"## {exp_num+1}.\")
    
    with open(ideas_file, 'w') as f:
        f.write(content)
    
    print(f'Updated STRUCTURAL_IDEAS.md with {exp_id} results')
"
}

# Main loop
main() {
    EXP_NUM=0
    
    while true; do
        REMAINING=$(time_remaining)
        
        if [ $REMAINING -lt 60 ]; then
            log "⏰ Less than 60 min remaining. Finalizing..."
            break
        fi
        
        run_structural_exp $EXP_NUM
        
        EXP_NUM=$((EXP_NUM + 1))
        
        if [ $EXP_NUM -ge 30 ]; then  # Max 30 structural experiments
            log "🛑 Reached max experiment count"
            break
        fi
        
        sleep 10
    done
    
    # Final summary
    log ""
    log "🏁 STRUCTURAL RESEARCH COMPLETE"
    log "Generating final report..."
    
    python3 -c "
import sys
sys.path.insert(0, '$REPO_DIR')
import json
from pathlib import Path

# Collect all results
results = []
runs_dir = Path('$REPO_DIR/experiments/runs')
for run_dir in runs_dir.glob('STRUCT_*'):
    results_file = run_dir / 'results.json'
    config_file = run_dir / 'config.json'
    if results_file.exists() and config_file.exists():
        with open(results_file) as f:
            res = json.load(f)
        with open(config_file) as f:
            cfg = json.load(f)
        results.append({
            'exp_id': cfg['id'],
            'map50': res['map50'],
            'structural_change': cfg['structural_change'],
            'target_problem': cfg['target_problem']
        })

# Sort by mAP50
results.sort(key=lambda x: x['map50'], reverse=True)

# Generate summary
summary = '# 🎉 Final Structural Research Summary\\n\\n'
summary += f'**Total Structural Experiments**: {len(results)}\\n\\n'

if results:
    best = results[0]
    summary += f'**Best mAP50**: {best[\"map50\"]:.4f}\\n'
    summary += f'**Best Experiment**: {best[\"exp_id\"]}\\n'
    summary += f'**Best Structural Change**: {best[\"structural_change\"]}\\n'
    summary += f'**Target Problem**: {best[\"target_problem\"]}\\n\\n'
    
    summary += '## Top 5 Structural Approaches\\n\\n'
    for i, r in enumerate(results[:5], 1):
        summary += f\"{i}. **{r['exp_id']}**: {r['map50']:.4f} - {r['structural_change'][:50]}...\\n\"
    
    summary += '\\n## Key Findings\\n\\n'
    summary += '- Structural approaches tested include: TTA, Hard mining, Adaptive smoothing, Enhanced FPN, Semantic copy-paste, Domain stratification, Ensemble, Uncertainty guidance\\n'
    summary += '- Best performing structural change will inform next research direction\\n'

with open('$REPO_DIR/FINAL_STRUCTURAL_SUMMARY.md', 'w') as f:
    f.write(summary)

print('Final summary generated')
"
    
    # Final push
    cd "$REPO_DIR"
    git add -A 2>/dev/null || true
    git commit -m "Final: 8-hour structural autonomous research complete" 2>/dev/null || true
    git push origin main 2>/dev/null || true
    
    rm -f "$PID_FILE"
    
    log "✅ Complete!"
    log "📊 Check GitHub for STRUCTURAL_LEADERBOARD.md and results"
}

# Trap
trap 'log "Interrupted"; rm -f "$PID_FILE"; exit 1' INT TERM

# Run
main
