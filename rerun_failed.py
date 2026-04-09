#!/usr/bin/env python3
"""Rerun only the failed experiments: NOVEL_012 (and skip NOVEL_018/019 if already done)."""
import sys
import subprocess
sys.path.insert(0, "/workspace/Hermes-YOLO")

import shutil
from pathlib import Path

BASE_DIR = Path("/workspace/Hermes-YOLO")
RUNS_DIR = BASE_DIR / "runs/detect/experiments/experiments/runs"

# Only rerun NOVEL_012 (018 and 019 already completed)
RERUN_IDS = ["NOVEL_012"]

# Force re-run by deleting incomplete run dirs (no results.csv)
for exp_id in RERUN_IDS:
    d = RUNS_DIR / exp_id
    if d.exists() and not (d / "results.csv").exists():
        print(f"[Setup] Removing incomplete dir: {d}")
        shutil.rmtree(d)
    elif d.exists() and (d / "results.csv").exists():
        print(f"[Setup] {exp_id} already has results.csv — skipping")
        RERUN_IDS = [x for x in RERUN_IDS if x != exp_id]

# Import runner
from novel_runner import run_experiment, update_leaderboard, generate_chart, EXPERIMENTS

failed_configs = [e for e in EXPERIMENTS if e["id"] in RERUN_IDS]

if not failed_configs:
    print("Nothing to rerun.")
    sys.exit(0)

print(f"\n{'='*60}")
print(f"Rerunning {len(failed_configs)} experiments: {[e['id'] for e in failed_configs]}")
print(f"{'='*60}\n")

results = {}
for config in failed_configs:
    result = run_experiment(config)
    results[config["id"]] = result
    try:
        update_leaderboard(config["id"], result, config)
    except Exception as e:
        print(f"[Leaderboard] Update failed: {e}")
    try:
        generate_chart()
    except Exception as e:
        print(f"[Chart] Failed: {e}")
    # Push directly via subprocess (PAT embedded in origin URL)
    try:
        subprocess.run(["git", "add", "-A",
                        "IDEA.md", "LEADERBOARD.md", "README.md",
                        "experiments/visualizations/"],
                       cwd=str(BASE_DIR), check=False, capture_output=True)
        subprocess.run(["git", "add",
                        f"runs/detect/experiments/experiments/runs/{config['id']}/results.csv"],
                       cwd=str(BASE_DIR), check=False, capture_output=True)
        msg = f"results: {config['id']} mAP50={result.get('map50', 0):.4f}"
        commit = subprocess.run(["git", "commit", "-m", msg, "--allow-empty"],
                                cwd=str(BASE_DIR), capture_output=True, text=True)
        push = subprocess.run(["git", "push", "origin", "main"],
                               cwd=str(BASE_DIR), capture_output=True, text=True)
        if push.returncode == 0:
            print(f"[Git] Pushed: {msg}")
        else:
            print(f"[Git] Push failed: {push.stderr.strip()}")
    except Exception as e:
        print(f"[Git] Error: {e}")

print(f"\n{'='*60}")
print("RERUN COMPLETE")
for eid, res in results.items():
    print(f"  {eid}: mAP50={res.get('map50', 0):.4f} | Recall={res.get('recall', 0):.4f}")
print(f"{'='*60}")
