#!/usr/bin/env python3
"""
Generate paper-ready tables from experiment results.

Reads all result JSON files and produces:
  1. LaTeX tables (copy-paste into paper)
  2. CSV summaries
  3. Console-formatted tables for quick review

Usage:
    python generate_tables.py              # all tables
    python generate_tables.py --table 1    # Table I only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import confidence_interval, cohens_d


def load_all_results(results_dir: str) -> dict:
    """Load all experiment results into a nested dict."""
    results = {}
    for path in Path(results_dir).rglob("*.json"):
        with open(path) as f:
            data = json.load(f)
        exp_name = data.get("experiment_name", path.parent.name)
        if exp_name not in results:
            results[exp_name] = []
        results[exp_name].append(data)
    return results


def generate_table_1(results: dict) -> str:
    """
    Table I: Open-Vocabulary Aerial Detection Results
    
    Paper location: Section V-A
    """
    header = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Open-Vocabulary Aerial Detection Results on VisDrone Test-Dev}\n"
        "\\label{tab:perception}\n\\small\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "\\textbf{Method} & \\textbf{Labels} & \\textbf{mAP@50} & "
        "\\textbf{mAP@50} & \\textbf{R@5} \\\\\n"
        " & & \\textbf{(Seen)} & \\textbf{(Novel)} & \\textbf{(Novel)} \\\\\n"
        "\\midrule\n"
    )

    rows = []

    # Try to load P1 results
    p1_dir = Path(results.get("results_root", "./results")) / "perception"
    model_results = {}

    for p1_file in p1_dir.glob("p1_*.json"):
        with open(p1_file) as f:
            data = json.load(f)
        model_name = data.get("model", p1_file.stem)
        model_results[model_name] = data

    if not model_results:
        # No results yet — return template with placeholders
        rows = [
            "YOLOv8-L (closed) & Full & \\tbd{XX.X} & N/A & N/A \\\\",
            "\\midrule",
            "CLIP ViT-L/14 & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "OWLv2 (base) & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "OWLv2 (large) & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "GD Swin-T & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "GD Swin-B & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "\\midrule",
            "OVP+S-T ($\\tau$=0.7) & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "OVP+S-T+Temporal & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "OVP+Active (5v) & 0 & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
            "\\textbf{OVAC (full)} & \\textbf{0} & \\tbd{XX.X} & \\tbd{XX.X} & \\tbd{XX.X} \\\\",
        ]
    else:
        # Real results!
        for name, data in sorted(model_results.items()):
            seen = data.get("seen_mAP50", 0) * 100
            novel = data.get("novel_mAP50", 0) * 100
            r5 = data.get("recall_at_k", {}).get("R@5", 0) * 100
            rows.append(f"{name} & 0 & {seen:.1f} & {novel:.1f} & {r5:.1f} \\\\")

    footer = (
        "\\bottomrule\n\\end{tabular}\n"
        "\\vspace{-0.5em}\n\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def generate_table_3(results: dict) -> str:
    """
    Table III: SAR Coordination Results
    
    Paper location: Section V-B
    """
    header = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{SAR Coordination Results (100 Trials, 4 Drones, Mean $\\pm$ 95\\% CI)}\n"
        "\\label{tab:coordination}\n\\small\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "\\textbf{Configuration} & \\textbf{MSR} & \\textbf{TTC} & "
        "\\textbf{Safety} & \\textbf{Verifier} \\\\\n"
        " & \\textbf{(\\%)} & \\textbf{(steps)} & "
        "\\textbf{Viol.} & \\textbf{Rej. (\\%)} \\\\\n"
        "\\midrule\n"
    )

    conditions = [
        ("Voronoi Coverage", "voronoi_coverage"),
        ("Frontier Exploration", "frontier_exploration"),
        ("Symbolic (grid)", "symbolic_grid"),
        ("LLM Only", "llm_only"),
        ("LLM + Rules Only", "llm_rules"),
        ("LLM + CBF Shield", "llm_cbf"),
        ("LLM + Hybrid Verifier", "llm_hybrid"),
        ("\\textbf{Full OVAC}", "full_ovac"),
    ]

    rows = []
    # Check for coordination results
    coord_dir = Path(results.get("results_root", "./results")) / "coordination"
    has_results = False

    for display_name, cond_id in conditions:
        result_file = coord_dir / f"c4_{cond_id}.json"
        if result_file.exists():
            has_results = True
            with open(result_file) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            msr_mean, msr_ci = metrics.get("msr_mean", 0), metrics.get("msr_ci", 0)
            ttc_mean = metrics.get("ttc_mean", 0)
            safety = metrics.get("safety_violations_mean", 0)
            rej = metrics.get("verifier_rejection_rate", 0)

            rej_str = f"{rej:.0f}" if rej > 0 else "N/A"
            rows.append(
                f"{display_name} & {msr_mean:.0f}$\\pm${msr_ci:.0f} & "
                f"{ttc_mean:.0f} & {safety:.1f} & {rej_str} \\\\"
            )
        else:
            # Placeholder
            rows.append(
                f"{display_name} & \\tbd{{XX$\\pm$X}} & "
                f"\\tbd{{XXXX}} & \\tbd{{X.X}} & \\tbd{{XX}} \\\\"
            )

    # Add section dividers
    rows.insert(3, "\\midrule")  # After classical baselines

    footer = (
        "\\bottomrule\n\\end{tabular}\n"
        "\\vspace{-0.5em}\n\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def console_summary(results_dir: str):
    """Print a quick console summary of all available results."""
    print(f"\n{'='*70}")
    print(f"  OVAC Results Summary")
    print(f"  Results directory: {results_dir}")
    print(f"{'='*70}\n")

    # Perception
    perc_dir = Path(results_dir) / "perception"
    if perc_dir.exists():
        print("Track A: Perception")
        print("-" * 50)
        for f in sorted(perc_dir.glob("p1_*.json")):
            with open(f) as fh:
                data = json.load(fh)
            model = data.get("model", f.stem)
            seen = data.get("seen_mAP50", 0)
            novel = data.get("novel_mAP50", 0)
            print(f"  {model:<30s} Seen: {seen:.4f}  Novel: {novel:.4f}")
        print()

    # Coordination
    coord_dir = Path(results_dir) / "coordination"
    if coord_dir.exists():
        print("Track B: Coordination")
        print("-" * 50)
        for f in sorted(coord_dir.glob("c4_*.json")):
            with open(f) as fh:
                data = json.load(fh)
            cond = data.get("condition", f.stem)
            msr = data.get("metrics", {}).get("msr_mean", 0)
            safety = data.get("metrics", {}).get("safety_violations_mean", 0)
            print(f"  {cond:<30s} MSR: {msr:.1f}%  Safety: {safety:.1f}")
        print()

    print(f"{'='*70}")
    print("To generate LaTeX tables: python generate_tables.py --latex")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--table", type=int, default=None, help="Generate specific table")
    args = parser.parse_args()

    results = {"results_root": args.results_dir}

    if args.latex:
        if args.table is None or args.table == 1:
            print("\n% TABLE I: Perception Results")
            print(generate_table_1(results))
        if args.table is None or args.table == 3:
            print("\n% TABLE III: Coordination Results")
            print(generate_table_3(results))
    else:
        console_summary(args.results_dir)


if __name__ == "__main__":
    main()
