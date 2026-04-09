#!/usr/bin/env python3
"""
STATUS CHECKER - Monitor all training progress
"""

import os
import json
from pathlib import Path
import subprocess

os.chdir("/workspace/Hermes-YOLO")

def check_status():
    print("🔍 HERMES-YOLO STATUS CHECK")
    print("="*60)
    
    # Check GPU
    print("\n📊 GPU Status:")
    result = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader", 
                          shell=True, capture_output=True, text=True)
    print(f"   {result.stdout.strip()}")
    
    # Check running processes
    print("\n🔴 Running Training:")
    result = subprocess.run("ps aux | grep 'train_' | grep python | grep -v grep | wc -l", 
                          shell=True, capture_output=True, text=True)
    running = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    print(f"   Active training processes: {running}")
    
    # Check completed experiments
    runs_dir = Path("experiments/runs")
    completed = 0
    best_map50 = 0
    best_exp = ""
    
    for exp_dir in runs_dir.glob("*"):
        results_file = exp_dir / "results.json"
        if results_file.exists():
            completed += 1
            with open(results_file) as f:
                data = json.load(f)
            map50 = data.get('map50', 0)
            if map50 > best_map50:
                best_map50 = map50
                best_exp = exp_dir.name
    
    print(f"\n✅ Completed Experiments: {completed}")
    print(f"🏆 Best Result: {best_exp} at {best_map50:.4f} mAP50")
    
    # Check BREAK_101 progress
    log_file = Path("/tmp/training_101.log")
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
        # Find last epoch info
        for line in reversed(lines):
            if "Epoch" in line and "/100" in line:
                print(f"\n📝 BREAK_101 Latest: {line.strip()}")
                break
    
    # Git status
    result = subprocess.run("git log --oneline | wc -l", shell=True, capture_output=True, text=True)
    commits = result.stdout.strip()
    print(f"\n📤 GitHub Commits: {commits}")
    
    print("\n" + "="*60)
    print("🔄 Autonomous research continuing...")
    print("🛑 Say 'STOP' to halt")

if __name__ == "__main__":
    check_status()
