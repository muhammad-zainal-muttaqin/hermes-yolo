#!/usr/bin/env python3
"""
Monitor NOVEL experiments and update IDEA.md + push to GitHub.
Runs in parallel with novel_runner.py to fix tracking.
"""

import os
import csv
import subprocess
import time
import re
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("/workspace/Hermes-YOLO")
CHART_DIR = BASE_DIR / "experiments/visualizations"
IDEA_MD = BASE_DIR / "IDEA.md"
LEADERBOARD_MD = BASE_DIR / "LEADERBOARD.md"

# All possible result locations
RESULT_SEARCH_ROOTS = [
    BASE_DIR / "runs/detect/experiments/runs",
    BASE_DIR / "runs/detect/experiments/experiments/runs",
    BASE_DIR / "runs/detect/experiments",
]

EXPERIMENT_IDS = [
    "NOVEL_001", "NOVEL_002", "NOVEL_003", "NOVEL_004", "NOVEL_005",
    "NOVEL_006", "NOVEL_007", "NOVEL_008", "NOVEL_009", "NOVEL_010",
]

EXPERIMENT_NAMES = {
    "NOVEL_001": "Label Smoothing + CosLR",
    "NOVEL_002": "L*a*b* Color Space Input",
    "NOVEL_003": "P2 Detection Head",
    "NOVEL_004": "SORD Ordinal Loss (σ=0.8)",
    "NOVEL_005": "Higher Resolution 768px",
    "NOVEL_006": "SORD + Label Smoothing",
    "NOVEL_007": "L*a*b* + P2 Head",
    "NOVEL_008": "SORD + P2 Head",
    "NOVEL_009": "Full Tier 1: LAB+SORD+P2",
    "NOVEL_010": "SORD Tight (σ=0.5)",
}

tracked = set()  # already processed


def find_results_csv(exp_id):
    for root in RESULT_SEARCH_ROOTS:
        candidates = list(root.glob(f"*{exp_id}*/results.csv")) + \
                     list(root.glob(f"*{exp_id}*/train/results.csv"))
        if candidates:
            return candidates[0]
    return None


def parse_results_csv(csv_path):
    best = {"map50": 0, "recall": 0, "precision": 0, "best_epoch": 0, "total_epochs": 0}
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return best
        best["total_epochs"] = len(rows)
        for i, row in enumerate(rows):
            row = {k.strip(): v.strip() for k, v in row.items()}
            map50_keys = [k for k in row if "map50" in k.lower() and "95" not in k.lower()]
            recall_keys = [k for k in row if "recall" in k.lower()]
            prec_keys = [k for k in row if "precision" in k.lower()]
            if map50_keys:
                v = float(row[map50_keys[0]])
                if v > best["map50"]:
                    best["map50"] = v
                    best["best_epoch"] = i + 1
                    if recall_keys:
                        best["recall"] = float(row[recall_keys[0]])
                    if prec_keys:
                        best["precision"] = float(row[prec_keys[0]])
    except Exception as e:
        print(f"  Parse error: {e}")
    return best


def count_epochs_done(csv_path):
    try:
        with open(csv_path) as f:
            return sum(1 for _ in f) - 1  # minus header
    except:
        return 0


def is_training_done(exp_id):
    """Check if training completed (results.csv has expected epoch count)."""
    csv_path = find_results_csv(exp_id)
    if not csv_path:
        return False
    # Check if training for this experiment is complete
    # We expect 15 epochs (or check for the lock/pid file)
    n = count_epochs_done(csv_path)
    return n >= 14  # at least 14 epochs done (might be 15 or more)


def update_idea_md(results_dict):
    """Update IDEA.md results table with all collected results."""
    content = IDEA_MD.read_text()

    for exp_id, r in results_dict.items():
        name = EXPERIMENT_NAMES.get(exp_id, exp_id)
        row_old = f"| {exp_id} |"

        new_row = (f"| {exp_id} | {name} | **{r['map50']:.4f}** | "
                   f"{r['recall']:.4f} | {r['precision']:.4f} | "
                   f"{r['best_epoch']}/{r['total_epochs']} | "
                   f"auto-tracked | {datetime.now().strftime('%Y-%m-%d')} | — |")

        # Replace existing row or add new one
        if row_old in content:
            lines = content.split("\n")
            updated = []
            for line in lines:
                if line.startswith(row_old):
                    updated.append(new_row)
                else:
                    updated.append(line)
            content = "\n".join(updated)
        else:
            # Append before closing of results table
            sentinel = "| — | BREAK_037 (hist best)"
            if sentinel in content:
                content = content.replace(
                    "\n---\n\n## Summary Progress",
                    f"\n{new_row}\n\n---\n\n## Summary Progress"
                )

    # Update NOVEL Best in summary
    if results_dict:
        best_id = max(results_dict, key=lambda k: results_dict[k]["map50"])
        best_val = results_dict[best_id]["map50"]
        content = re.sub(
            r"NOVEL Series Best \| .+? \|",
            f"NOVEL Series Best | **{best_val:.4f}** ({best_id}) |",
            content
        )

    IDEA_MD.write_text(content)
    print(f"  [IDEA.md] Updated with {len(results_dict)} results")


def update_leaderboard(results_dict):
    content = LEADERBOARD_MD.read_text()

    # Add NOVEL results section if not present
    if "## NOVEL Series Results" not in content:
        novel_section = "\n\n## NOVEL Series Results (15-epoch scout runs)\n\n| Rank | Experiment | mAP50 | Recall | Epochs | Strategy |\n|:----:|:-----------|:-----:|:------:|:------:|:---------|\n"
        content += novel_section

    # Update or add each result
    for exp_id, r in sorted(results_dict.items(), key=lambda x: -x[1]["map50"]):
        row = f"| - | {exp_id} | {r['map50']:.4f} | {r['recall']:.4f} | {r['best_epoch']}/{r['total_epochs']} | {EXPERIMENT_NAMES.get(exp_id, '')} |"
        if exp_id in content:
            # Replace existing
            lines = content.split("\n")
            content = "\n".join([row if exp_id in l and "|" in l else l for l in lines])
        else:
            content += f"{row}\n"

    LEADERBOARD_MD.write_text(content)
    print("  [LEADERBOARD.md] Updated")


def generate_chart(results_dict):
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Historical results from old runs
    old_results = {
        "BREAK_005": 0.502, "BREAK_034": 0.5207, "BREAK_035": 0.5207,
        "BREAK_036": 0.5207, "BREAK_101": 0.5250,
    }

    # All results combined
    all_ids = list(old_results.keys()) + [k for k in EXPERIMENT_IDS if k in results_dict]
    all_vals = [old_results.get(k, results_dict.get(k, {}).get("map50", 0)) for k in all_ids]

    def color(exp_id):
        if exp_id.startswith("NOVEL"):
            return "#00ff88"
        elif "101" in exp_id:
            return "#ff4444"
        elif any(x in exp_id for x in ["034", "035", "036"]):
            return "#ff8800"
        return "#4488ff"

    colors = [color(k) for k in all_ids]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), facecolor="#0d1117")
    fig.suptitle("Hermes-YOLO — TBS Oil Palm Detection Progress (NOVEL Series)",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#30363d")
        ax.tick_params(colors="#8b949e")

    # Panel 1: All experiments
    ax1.bar(range(len(all_ids)), all_vals, color=colors, width=0.7, alpha=0.9)
    ax1.axhline(0.5298, color="#ff4444", ls="--", lw=1.5, alpha=0.7, label="Hist best (0.5298)")
    ax1.axhline(0.504, color="#888", ls=":", lw=1, alpha=0.7, label="Baseline (0.504)")
    ax1.axhline(0.70, color="#ffff00", ls="--", lw=1, alpha=0.5, label="Target (0.70)")

    if all_vals:
        bv = max(all_vals)
        bi = all_vals.index(bv)
        ax1.annotate(f"Best: {bv:.4f}", xy=(bi, bv), xytext=(bi, bv + 0.015),
                    color="white", fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1))

    ax1.set_xticks(range(len(all_ids)))
    ax1.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
    ax1.set_ylabel("mAP50 (B)", color="#8b949e")
    ax1.set_ylim(0.40, max(max(all_vals + [0]) + 0.06, 0.72))
    ax1.legend(facecolor="#161b22", labelcolor="white", fontsize=8, loc="upper left")
    ax1.set_title("All Experiments — mAP50 Comparison", color="#e6edf3", fontsize=11)

    # Panel 2: NOVEL series only
    novel_ids = [k for k in EXPERIMENT_IDS if k in results_dict]
    novel_vals = [results_dict[k]["map50"] for k in novel_ids]

    if novel_vals:
        bars = ax2.bar(range(len(novel_ids)), novel_vals, color="#00ff88", width=0.6, alpha=0.9)
        # Annotate each bar
        for i, (nid, nv) in enumerate(zip(novel_ids, novel_vals)):
            ax2.text(i, nv + 0.003, f"{nv:.4f}", ha="center", va="bottom",
                    color="white", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "NOVEL experiments in progress...",
                ha="center", va="center", color="#8b949e", fontsize=13,
                transform=ax2.transAxes)

    ax2.axhline(0.5250, color="#ff4444", ls="--", lw=1.5, alpha=0.7, label="BREAK_101 (0.5250)")
    ax2.axhline(0.5298, color="#ff8800", ls="--", lw=1.2, alpha=0.6, label="Hist best (0.5298)")
    ax2.axhline(0.70, color="#ffff00", ls="--", lw=1, alpha=0.5, label="Target (0.70)")

    if novel_ids:
        ax2.set_xticks(range(len(novel_ids)))
        ax2.set_xticklabels(novel_ids, rotation=30, ha="right", fontsize=9, color="#8b949e")
    ax2.set_ylabel("mAP50 (B)", color="#8b949e")
    ax2.set_ylim(0.40, 0.76)
    ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8, loc="upper right")
    ax2.set_title("NOVEL Series — Novel Strategy Comparison (15 epochs each)", color="#e6edf3", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = CHART_DIR / "progress_map50.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Chart] Saved: {out}")


def git_push(msg):
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("  [Git] No token")
        return
    try:
        remote = f"https://{token}@github.com/muhammad-zainal-muttaqin/Hermes-YOLO.git"
        subprocess.run(["git", "remote", "set-url", "origin", remote],
                      cwd=str(BASE_DIR), check=True, capture_output=True)
        subprocess.run(["git", "add", "IDEA.md", "LEADERBOARD.md",
                        "experiments/visualizations/progress_map50.png"],
                      cwd=str(BASE_DIR), capture_output=True)
        r = subprocess.run(["git", "commit", "-m", msg],
                          cwd=str(BASE_DIR), capture_output=True, text=True)
        if "nothing to commit" in r.stdout:
            return
        subprocess.run(["git", "push", "origin", "main"],
                      cwd=str(BASE_DIR), check=True, capture_output=True)
        print(f"  [Git] Pushed: {msg[:60]}")
    except Exception as e:
        print(f"  [Git] Error: {e}")


def main():
    print("=" * 50)
    print("NOVEL Experiment Monitor")
    print(f"Tracking: {', '.join(EXPERIMENT_IDS)}")
    print("=" * 50)

    results = {}
    last_push = set()

    # Load already-completed results
    for exp_id in EXPERIMENT_IDS:
        csv_path = find_results_csv(exp_id)
        if csv_path and is_training_done(exp_id):
            r = parse_results_csv(csv_path)
            if r["map50"] > 0.1:
                results[exp_id] = r
                print(f"  [Loaded] {exp_id}: mAP50={r['map50']:.4f}")

    # Watch for new completions
    check_interval = 30  # seconds
    max_wait = 7200      # 2 hours max wait
    elapsed = 0

    while elapsed < max_wait:
        changed = False

        for exp_id in EXPERIMENT_IDS:
            if exp_id in last_push:
                continue

            csv_path = find_results_csv(exp_id)
            if not csv_path:
                continue

            n_epochs = count_epochs_done(csv_path)

            if is_training_done(exp_id) and exp_id not in results:
                r = parse_results_csv(csv_path)
                if r["map50"] > 0.01:
                    results[exp_id] = r
                    print(f"\n  ✅ {exp_id} DONE: mAP50={r['map50']:.4f} | "
                          f"Recall={r['recall']:.4f} | Epoch={r['best_epoch']}/{r['total_epochs']}")
                    changed = True
            elif csv_path and n_epochs > 0:
                # Show progress
                r_current = parse_results_csv(csv_path)
                print(f"  [{exp_id}] epoch {n_epochs}/15 | "
                      f"mAP50={r_current['map50']:.4f}", end="\r")

        if changed and results:
            print(f"\n  Updating tracking files ({len(results)} experiments done)...")
            update_idea_md(results)
            update_leaderboard(results)
            generate_chart(results)

            done_ids = [k for k in results]
            best_id = max(results, key=lambda k: results[k]["map50"])
            best_val = results[best_id]["map50"]
            msg = f"track: {', '.join(done_ids[-3:])} — best={best_id} mAP50={best_val:.4f}"
            git_push(msg)
            last_push.update(done_ids)

        # Check if all done
        if all(is_training_done(exp_id) for exp_id in EXPERIMENT_IDS):
            # Final update
            for exp_id in EXPERIMENT_IDS:
                csv_path = find_results_csv(exp_id)
                if csv_path and exp_id not in results:
                    r = parse_results_csv(csv_path)
                    if r["map50"] > 0.01:
                        results[exp_id] = r

            print("\n\n" + "=" * 50)
            print("ALL NOVEL EXPERIMENTS COMPLETE!")
            print("=" * 50)
            for exp_id in EXPERIMENT_IDS:
                if exp_id in results:
                    r = results[exp_id]
                    print(f"  {exp_id}: mAP50={r['map50']:.4f} | {EXPERIMENT_NAMES.get(exp_id, '')}")

            if results:
                best_id = max(results, key=lambda k: results[k]["map50"])
                print(f"\nBEST: {best_id} → mAP50={results[best_id]['map50']:.4f}")

            update_idea_md(results)
            update_leaderboard(results)
            generate_chart(results)
            git_push(f"final: all NOVEL series complete — best mAP50={results.get(best_id, {}).get('map50', 0):.4f}")
            break

        time.sleep(check_interval)
        elapsed += check_interval

    print("Monitor done.")


if __name__ == "__main__":
    main()
