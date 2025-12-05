#!/usr/bin/env python3
"""
simulate_bkt.py

Run BKT simulations for N students and M questions each, collect mastery logs,
produce CSVs and plots showing P(K) trajectories and summary metrics.

Usage (example):
    source venv/bin/activate
    python scripts/simulate_bkt.py --n_students 50 --n_steps 200 --n_concepts 10 --out_dir reports/bkt_sim_1 --seed 42

Outputs:
 - <out_dir>/mastery_timeseries.csv   : per-step mastery snapshots
 - <out_dir>/summary.csv              : per-concept summary metrics
 - <out_dir>/plots/pk_trajectories.png
 - <out_dir>/plots/pk_mean_trajectories.png
"""
from __future__ import annotations
import os
import csv
import argparse
import random
import statistics
from collections import defaultdict
from datetime import datetime
import math

import matplotlib.pyplot as plt

# Import your BKT update function
from Model.bkt import bkt_update

DEFAULT_PARAMS = {
    "p_trans": 0.15,
    "p_guess": 0.2,
    "p_slip": 0.1,
}

def simulate_one_run(n_students: int, n_steps: int, n_concepts: int, params_per_concept: dict, seed: int = None):
    """
    Simulate n_students. Each student starts with prior p_mastery=0.05 for all concepts.
    At each step a random concept is chosen. Observed correctness is sampled using:
      p_observed_correct = p_mastery * (1 - p_slip) + (1 - p_mastery) * p_guess
    Then bkt_update is applied, and we record old/new p_mastery.
    Returns:
      - logs: list of dicts: {student_id, step, concept_id, old_p, new_p, correct}
      - per_student_final_mastery: dict student -> {concept: p}
    """
    if seed is not None:
        random.seed(seed)

    logs = []
    final_mastery = {}

    # initialize per-student mastery
    for s in range(n_students):
        student_id = f"student_{s+1:03d}"
        # each concept has its own params; if not provided use DEFAULT_PARAMS
        mastery = {f"c{c+1}": 0.05 for c in range(n_concepts)}
        final_mastery[student_id] = mastery

    for step in range(1, n_steps + 1):
        for s in range(n_students):
            student_id = f"student_{s+1:03d}"
            # pick a random concept (uniform)
            concept_idx = random.randrange(n_concepts)
            concept_id = f"c{concept_idx+1}"
            old_p = final_mastery[student_id][concept_id]
            params = params_per_concept.get(concept_id, DEFAULT_PARAMS)
            p_slip = params.get("p_slip", DEFAULT_PARAMS["p_slip"])
            p_guess = params.get("p_guess", DEFAULT_PARAMS["p_guess"])

            # simulate observed correctness
            p_obs = old_p * (1.0 - p_slip) + (1.0 - old_p) * p_guess
            correct = random.random() < p_obs

            # apply the BKT update (assumed deterministic function)
            try:
                new_p = bkt_update(old_p, correct, params)
            except Exception as e:
                # if bkt_update throws, fallback to a safe calculation:
                new_p = max(0.0, min(1.0, old_p + (0.1 if correct else -0.05)))

            # record
            logs.append({
                "student_id": student_id,
                "step": step,
                "concept_id": concept_id,
                "old_p": old_p,
                "new_p": new_p,
                "correct": int(bool(correct)),
            })

            final_mastery[student_id][concept_id] = new_p

    return logs, final_mastery

def aggregate_time_series(logs, n_concepts):
    """
    Convert logs into per-step mean mastery per concept and overall mean.
    Returns dict: concept_id -> list of (step, mean_p)
    and overall_mean -> list of (step, mean_p_over_all_concepts)
    """
    # Organize by step and concept
    step_concept_latest = defaultdict(lambda: {})  # step -> concept -> last p at that step (average across students)
    # We'll compute for each step the average new_p across students for each concept
    step_max = max(l["step"] for l in logs) if logs else 0

    concept_steps = {f"c{c+1}": [None] * (step_max + 1) for c in range(n_concepts)}

    for step in range(1, step_max + 1):
        # filter logs of this step
        entries = [l for l in logs if l["step"] == step]
        # group by concept
        by_concept = defaultdict(list)
        for e in entries:
            by_concept[e["concept_id"]].append(e["new_p"])
        for cid, values in by_concept.items():
            meanp = statistics.mean(values) if values else None
            concept_steps[cid][step] = meanp

    # fill forward None values with last known
    for cid, arr in concept_steps.items():
        last = None
        for i in range(1, len(arr)):
            if arr[i] is None:
                arr[i] = last
            else:
                last = arr[i]

    # Overall mean across concepts per step
    overall = [None] * (step_max + 1)
    for step in range(1, step_max + 1):
        vals = []
        for cid in concept_steps.keys():
            v = concept_steps[cid][step]
            if v is not None:
                vals.append(v)
        overall[step] = statistics.mean(vals) if vals else None

    return concept_steps, overall

def save_logs_csv(logs, out_file):
    keys = ["student_id", "step", "concept_id", "old_p", "new_p", "correct"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in logs:
            w.writerow(r)

def save_summary_csv(final_mastery, out_file):
    """
    final_mastery: dict student -> {concept: p}
    produce per-concept average final p
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # flatten per concept
    per_concept = defaultdict(list)
    for s, cmap in final_mastery.items():
        for cid, p in cmap.items():
            per_concept[cid].append(p)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_id", "mean_final_p", "median_final_p", "n_students"])
        for cid, vals in sorted(per_concept.items()):
            writer.writerow([cid, statistics.mean(vals), statistics.median(vals), len(vals)])

def plot_trajectories(concept_steps, overall, out_png, top_k: int = 6):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # plot mean trajectories for top_k concepts (by final value)
    # compute final values
    final_vals = [(cid, arr[-1]) for cid, arr in concept_steps.items()]
    final_sorted = sorted(final_vals, key=lambda x: (x[1] if x[1] is not None else -1))
    # pick first top_k lowest final mastery to show trajectories
    pick = [cid for cid, _ in final_sorted[:top_k]]
    plt.figure(figsize=(10, 6))
    for cid in pick:
        arr = concept_steps[cid]
        steps = list(range(len(arr)))
        plt.plot(steps[1:], arr[1:], label=cid)  # skip index 0 (unused)
    plt.xlabel("Step")
    plt.ylabel("Mean p_mastery")
    plt.title("Mean P(K) trajectories (selected concepts)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_overall(overall, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    steps = list(range(len(overall)))
    plt.figure(figsize=(8,5))
    plt.plot(steps[1:], overall[1:])
    plt.xlabel("Step")
    plt.ylabel("Mean p_mastery (overall)")
    plt.title("Overall Mean P(K) over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def compute_metrics(logs):
    """
    Compute average mastery change after correct vs incorrect.
    Return dict with mean_delta_correct, mean_delta_incorrect
    """
    deltas_correct = []
    deltas_incorrect = []
    for e in logs:
        delta = e["new_p"] - e["old_p"]
        if e["correct"]:
            deltas_correct.append(delta)
        else:
            deltas_incorrect.append(delta)
    return {
        "mean_delta_correct": statistics.mean(deltas_correct) if deltas_correct else 0.0,
        "mean_delta_incorrect": statistics.mean(deltas_incorrect) if deltas_incorrect else 0.0,
        "n_correct": len(deltas_correct),
        "n_incorrect": len(deltas_incorrect),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--n_concepts", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="reports/bkt_sim")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # create simple concept params dict (could be varied per concept)
    params_per_concept = {}
    for i in range(args.n_concepts):
        cid = f"c{i+1}"
        params_per_concept[cid] = DEFAULT_PARAMS.copy()

    logs, final_mastery = simulate_one_run(
        n_students=args.n_students,
        n_steps=args.n_steps,
        n_concepts=args.n_concepts,
        params_per_concept=params_per_concept,
        seed=args.seed,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    # save logs
    save_logs_csv(logs, os.path.join(out_dir, "mastery_timeseries.csv"))
    save_summary_csv(final_mastery, os.path.join(out_dir, "summary.csv"))

    concept_steps, overall = aggregate_time_series(logs, args.n_concepts)
    plots_dir = os.path.join(out_dir, "plots")
    plot_trajectories(concept_steps, os.path.join(plots_dir, "pk_trajectories.png"))
    plot_overall(overall, os.path.join(plots_dir, "pk_overall.png"))

    metrics = compute_metrics(logs)
    # save metrics
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # print summary to console
    print("Simulation complete.")
    print("Outdir:", out_dir)
    print("Metrics:", metrics)
    print("Plots:", os.path.join(plots_dir, "pk_trajectories.png"), os.path.join(plots_dir, "pk_overall.png"))

if __name__ == "__main__":
    main()
