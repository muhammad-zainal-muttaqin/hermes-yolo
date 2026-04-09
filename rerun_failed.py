#!/usr/bin/env python3
"""Rerun only the 3 failed experiments: NOVEL_012, NOVEL_018, NOVEL_019."""
import sys
sys.path.insert(0, "/workspace/Hermes-YOLO")

import shutil
from pathlib import Path

# Force re-run by deleting incomplete run dirs
RUNS_DIR = Path("/workspace/Hermes-YOLO/runs/detect/experiments/experiments/runs")
for exp_id in ["NOVEL_012", "NOVEL_018", "NOVEL_019"]:
    d = RUNS_DIR / exp_id
    if d.exists() and not (d / "results.csv").exists():
        print(f"[Setup] Removing incomplete dir: {d}")
        shutil.rmtree(d)

# Import runner and run only failed experiments
from novel_runner import run_experiment, update_leaderboard, generate_chart, git_push, EXPERIMENTS

failed_ids = {"NOVEL_012", "NOVEL_018", "NOVEL_019"}
failed_configs = [e for e in EXPERIMENTS if e["id"] in failed_ids]

print(f"\n{'='*60}")
print(f"Rerunning {len(failed_configs)} failed experiments: {[e['id'] for e in failed_configs]}")
print(f"{'='*60}\n")

results = {}
for config in failed_configs:
    result = run_experiment(config)
    results[config["id"]] = result
    try:
        update_leaderboard(config["id"], config["name"], result, config["epochs"])
    except Exception as e:
        print(f"[Leaderboard] Update failed: {e}")
    try:
        generate_chart()
    except Exception as e:
        print(f"[Chart] Failed: {e}")
    try:
        git_push(f"NOVEL rerun: {config['id']} mAP50={result.get('map50', 0):.4f}")
    except Exception as e:
        print(f"[Git] Push failed: {e}")

print(f"\n{'='*60}")
print("RERUN COMPLETE")
for eid, res in results.items():
    print(f"  {eid}: mAP50={res.get('map50', 0):.4f} | Recall={res.get('recall', 0):.4f}")
print(f"{'='*60}")
