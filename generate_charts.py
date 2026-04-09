#!/usr/bin/env python3
"""
Generate charts for Hermes-YOLO analysis
"""

import os
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir("/workspace/Hermes-YOLO")

# Load all experiment results
runs_dir = Path("experiments/runs")
all_results = []

for exp_dir in sorted(runs_dir.glob("*")):
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        data['exp_id'] = exp_dir.name
        all_results.append(data)

# Sort by mAP50
all_results.sort(key=lambda x: x.get('map50', 0), reverse=True)

print(f"📊 Generating charts for {len(all_results)} experiments...")

# Chart 1: Progress over time
fig, ax = plt.subplots(figsize=(14, 6))
exp_nums = []
map50s_time = []
for r in all_results:
    if '_' in r['exp_id']:
        try:
            num = int(r['exp_id'].split('_')[1])
            exp_nums.append(num)
            map50s_time.append(r.get('map50', 0))
        except:
            pass

ax.plot(exp_nums, map50s_time, 'o-', markersize=3, linewidth=1, alpha=0.6, color='steelblue')
ax.axhline(y=0.5225, color='red', linestyle='--', label='Baseline (0.5225)', linewidth=2)
ax.axhline(y=0.5375, color='green', linestyle='--', label='Best (0.5375)', linewidth=2)
ax.set_xlabel('Experiment Number', fontsize=12)
ax.set_ylabel('mAP50', fontsize=12)
ax.set_title(f'HERMES-YOLO: Progress Over {len(all_results)} Experiments', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/visualizations/progress_all_experiments.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiments/visualizations/progress_all_experiments.png")

# Chart 2: Distribution
fig, ax = plt.subplots(figsize=(10, 6))
map50_values = [r.get('map50', 0) for r in all_results if r.get('map50', 0) > 0]
ax.hist(map50_values, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(x=np.mean(map50_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(map50_values):.4f}')
ax.axvline(x=np.median(map50_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(map50_values):.4f}')
ax.axvline(x=0.5225, color='orange', linestyle=':', linewidth=2, label='Baseline')
ax.set_xlabel('mAP50', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'mAP50 Distribution (n={len(map50_values)})', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('experiments/visualizations/distribution_map50.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiments/visualizations/distribution_map50.png")

# Chart 3: Top 15 comparison
fig, ax = plt.subplots(figsize=(12, 8))
top_15 = all_results[:15]
exp_ids = [r['exp_id'] for r in top_15]
map50s = [r.get('map50', 0) for r in top_15]
colors = ['#2ecc71' if r['exp_id'].startswith('STRUCT') else '#3498db' for r in top_15]
bars = ax.barh(range(len(exp_ids)), map50s, color=colors, edgecolor='black')
ax.set_yticks(range(len(exp_ids)))
ax.set_yticklabels(exp_ids, fontsize=10)
ax.set_xlabel('mAP50', fontsize=12)
ax.set_title('Top 15 Experiments by mAP50', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.axvline(x=0.5225, color='red', linestyle='--', alpha=0.7, label='Baseline (0.5225)')
ax.legend(fontsize=10)
for i, (bar, val) in enumerate(zip(bars, map50s)):
    ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('experiments/visualizations/top15_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiments/visualizations/top15_comparison.png")

print(f"\n🎉 ALL CHARTS GENERATED!")
print(f"   Total experiments: {len(all_results)}")
print(f"   Best: {all_results[0]['exp_id']} at {all_results[0].get('map50', 0):.4f} mAP50")
