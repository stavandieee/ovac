#!/usr/bin/env python3
"""
OVAC Master Experiment Runner

Runs all experiments in dependency order with proper logging.
Each experiment saves results to results/{experiment_name}/.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --track A          # Perception only
    python run_all.py --track B          # Coordination only
    python run_all.py --experiment p1    # Single experiment
    python run_all.py --dry-run          # Print plan without running

Experiment dependency graph:
    P1 (baseline OVD) ─────────────────────────┐
    P2 (YOLOv8 baseline) ──────────────────────┤
    P3 (student-teacher) ── depends on P1 ─────┤── Track A results
    P4 (temporal fusion) ── depends on P1 ─────┤
    P5 (active re-obs) ─── depends on P1 ──────┘
                                                │
              calibrate perception sim ─────────┘
                                                │
    C1 (LLM plan quality) ─── no dependencies ─┤
    C2 (train translator) ─── no dependencies ──┤
    C3 (train verifier) ──── needs rollouts ────┤── Track B results
    C4 (SAR trials) ──────── needs C2, C3 ──────┤
    C5 (threshold sweep) ── needs C3 ───────────┤
    C6 (comms overhead) ─── post-process C4 ────┘
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import load_config, set_seed, ExperimentLogger


# ================================================================
# Experiment Registry
# ================================================================

EXPERIMENTS = {
    # Track A: Perception
    "p1": {
        "name": "P1: Baseline OVD Evaluation",
        "track": "A",
        "dependencies": [],
        "gpu_hours": 4.0,
        "script": "perception.eval.run_p1_baseline",
    },
    "p2": {
        "name": "P2: YOLOv8 Closed-Set Baseline",
        "track": "A",
        "dependencies": [],
        "gpu_hours": 8.0,
        "script": "perception.eval.run_p2_yolo",
    },
    "p3": {
        "name": "P3: Student-Teacher Adaptation",
        "track": "A",
        "dependencies": ["p1"],
        "gpu_hours": 12.0,
        "script": "perception.adaptation.run_p3_student_teacher",
    },
    "p4": {
        "name": "P4: Temporal Fusion",
        "track": "A",
        "dependencies": ["p1"],
        "gpu_hours": 6.0,
        "script": "perception.eval.run_p4_temporal",
    },
    "p5": {
        "name": "P5: Active Re-Observation",
        "track": "A",
        "dependencies": ["p1"],
        "gpu_hours": 4.0,
        "script": "perception.eval.run_p5_active",
    },
    # Track B: Coordination
    "c1": {
        "name": "C1: LLM Plan Quality",
        "track": "B",
        "dependencies": [],
        "gpu_hours": 0.5,
        "script": "coordination.scripts.run_c1_plan_quality",
    },
    "c2": {
        "name": "C2: Train Grounding Translator",
        "track": "B",
        "dependencies": [],
        "gpu_hours": 2.0,
        "script": "coordination.scripts.run_c2_translator",
    },
    "c3": {
        "name": "C3: Train Verifier",
        "track": "B",
        "dependencies": [],
        "gpu_hours": 1.0,
        "script": "coordination.scripts.run_c3_verifier",
    },
    "c4": {
        "name": "C4: SAR Trials (main result)",
        "track": "B",
        "dependencies": ["c2", "c3"],
        "gpu_hours": 2.0,
        "script": "coordination.scripts.run_c4_sar_trials",
    },
    "c5": {
        "name": "C5: Verifier Threshold Sweep",
        "track": "B",
        "dependencies": ["c3"],
        "gpu_hours": 1.0,
        "script": "coordination.scripts.run_c5_threshold",
    },
    "c6": {
        "name": "C6: Communication Overhead",
        "track": "B",
        "dependencies": ["c4"],
        "gpu_hours": 0.0,
        "script": "coordination.scripts.run_c6_comms",
    },
}


def topological_sort(experiments: dict, selected: list = None) -> list:
    """Sort experiments by dependency order."""
    if selected is None:
        selected = list(experiments.keys())

    # Include dependencies
    to_run = set()
    def add_with_deps(exp_id):
        if exp_id in to_run:
            return
        to_run.add(exp_id)
        for dep in experiments[exp_id]["dependencies"]:
            add_with_deps(dep)
    for eid in selected:
        add_with_deps(eid)

    # Topological sort
    order = []
    visited = set()
    def visit(eid):
        if eid in visited:
            return
        visited.add(eid)
        for dep in experiments[eid]["dependencies"]:
            visit(dep)
        order.append(eid)
    for eid in sorted(to_run):
        visit(eid)

    return order


def print_execution_plan(order: list, experiments: dict):
    """Print the execution plan."""
    total_gpu = sum(experiments[e]["gpu_hours"] for e in order)
    print(f"\n{'='*65}")
    print(f"  OVAC Experiment Execution Plan")
    print(f"  Estimated GPU time: {total_gpu:.1f} hours")
    print(f"{'='*65}")
    for i, eid in enumerate(order, 1):
        exp = experiments[eid]
        deps = ", ".join(exp["dependencies"]) if exp["dependencies"] else "none"
        print(f"  {i:2d}. [{eid.upper():3s}] {exp['name']:<40s} "
              f"({exp['gpu_hours']:.1f}h GPU, deps: {deps})")
    print(f"{'='*65}\n")


def check_results_exist(experiment_id: str, results_dir: str) -> bool:
    """Check if results already exist for an experiment."""
    result_dir = Path(results_dir) / experiment_id
    if result_dir.exists():
        results = list(result_dir.glob("*.json"))
        if results:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="OVAC Experiment Runner")
    parser.add_argument("--track", choices=["A", "B"], default=None,
                        help="Run only Track A or B")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run single experiment (e.g., p1, c4)")
    parser.add_argument("--config", type=str,
                        default="configs/master_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments with existing results")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed or config["project"]["seed"]

    # Determine which experiments to run
    if args.experiment:
        selected = [args.experiment.lower()]
        if selected[0] not in EXPERIMENTS:
            print(f"Unknown experiment: {selected[0]}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            sys.exit(1)
    elif args.track:
        selected = [k for k, v in EXPERIMENTS.items() if v["track"] == args.track]
    else:
        selected = list(EXPERIMENTS.keys())

    order = topological_sort(EXPERIMENTS, selected)
    print_execution_plan(order, EXPERIMENTS)

    if args.dry_run:
        print("Dry run — no experiments executed.")
        return

    # Run experiments
    results_dir = config["paths"]["results_root"]
    for exp_id in order:
        exp = EXPERIMENTS[exp_id]

        if args.skip_existing and check_results_exist(exp_id, results_dir):
            print(f"\n[SKIP] {exp_id.upper()}: Results already exist")
            continue

        print(f"\n{'#'*65}")
        print(f"  Running: {exp['name']}")
        print(f"  Estimated time: {exp['gpu_hours']:.1f} GPU hours")
        print(f"{'#'*65}\n")

        set_seed(seed)
        start = time.time()

        try:
            # Dynamic import and run
            module_path = exp["script"]
            parts = module_path.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            run_fn = getattr(module, parts[1])
            run_fn(config, seed)
        except (ImportError, AttributeError) as e:
            print(f"[WARNING] Could not import {exp['script']}: {e}")
            print(f"  This experiment script needs to be implemented.")
            print(f"  Expected function: {exp['script']}(config, seed)")
            continue
        except Exception as e:
            print(f"[ERROR] {exp_id.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - start
        print(f"\n✓ {exp_id.upper()} complete in {elapsed/60:.1f} minutes")

    print(f"\n{'='*65}")
    print(f"  All experiments complete!")
    print(f"  Results in: {results_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
