"""
OVAC Experiment Infrastructure

Handles: config loading, seeding, logging, result serialization.
Every experiment script imports this module — ensures consistency.
"""

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ================================================================
# Config
# ================================================================

def load_config(path: str = "configs/master_config.yaml") -> dict:
    """Load master config with environment variable expansion."""
    with open(path) as f:
        text = f.read()
    # Expand ${VAR:default} patterns
    import re
    def expand(match):
        var = match.group(1)
        default = match.group(2) if match.group(2) else ""
        return os.environ.get(var, default)
    text = re.sub(r'\$\{(\w+):?([^}]*)\}', expand, text)
    return yaml.safe_load(text)


# ================================================================
# Reproducibility
# ================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_environment_info() -> dict:
    """Capture environment for reproducibility metadata."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
    except ImportError:
        info["torch_version"] = "not installed"
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    return info


# ================================================================
# Result Logging
# ================================================================

@dataclass
class ExperimentResult:
    """Standardized result container for all experiments."""
    experiment_id: str
    experiment_name: str
    config_hash: str
    seed: int
    environment: dict
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    per_item_results: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save result as JSON with pretty printing."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logging.info(f"Results saved: {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentResult":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class ExperimentLogger:
    """Structured logger for experiment runs."""

    def __init__(self, experiment_name: str, config: dict, seed: int,
                 results_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.config = config
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.start_time = datetime.now(timezone.utc)

        # Create unique experiment ID
        config_str = json.dumps(config, sort_keys=True, default=str)
        self.config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.experiment_id = (
            f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
            f"_s{seed}_{self.config_hash}"
        )

        # Setup logging
        log_dir = self.results_dir / experiment_name / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{self.experiment_id}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        self.logger = logging.getLogger(experiment_name)

        # Initialize result container
        self.result = ExperimentResult(
            experiment_id=self.experiment_id,
            experiment_name=experiment_name,
            config_hash=self.config_hash,
            seed=seed,
            environment=get_environment_info(),
            start_time=self.start_time.isoformat(),
        )

        self.logger.info(f"Experiment: {self.experiment_id}")
        self.logger.info(f"Config hash: {self.config_hash}")
        self.logger.info(f"Seed: {seed}")

    def log_metric(self, key: str, value: float):
        """Log a single metric."""
        self.result.metrics[key] = value
        self.logger.info(f"METRIC {key}: {value}")

    def log_metrics(self, metrics: dict):
        """Log multiple metrics."""
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_item(self, item: dict):
        """Log a per-item result (e.g., per-image, per-trial)."""
        self.result.per_item_results.append(item)

    def finish(self) -> ExperimentResult:
        """Finalize and save results."""
        end_time = datetime.now(timezone.utc)
        self.result.end_time = end_time.isoformat()
        self.result.duration_seconds = (end_time - self.start_time).total_seconds()

        # Save
        result_path = (
            self.results_dir / self.experiment_name
            / f"{self.experiment_id}.json"
        )
        self.result.save(str(result_path))

        self.logger.info(
            f"Experiment complete in {self.result.duration_seconds:.1f}s"
        )
        return self.result


# ================================================================
# Metrics Utilities
# ================================================================

def compute_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(detections, ground_truth, iou_threshold=0.5):
    """
    Average Precision for one category across an image set.
    detections: list of {bbox, score}
    ground_truth: list of {bbox}
    """
    if not ground_truth:
        return 1.0 if not detections else 0.0
    if not detections:
        return 0.0

    dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    n_gt = len(ground_truth)
    matched = [False] * n_gt
    tp, fp = [], []

    for d in dets:
        best_iou, best_idx = 0, -1
        for gi, g in enumerate(ground_truth):
            iou = compute_iou(d["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_threshold and not matched[best_idx]:
            tp.append(1); fp.append(0); matched[best_idx] = True
        else:
            tp.append(0); fp.append(1)

    tp_c = np.cumsum(tp).astype(float)
    fp_c = np.cumsum(fp).astype(float)
    precision = tp_c / (tp_c + fp_c)
    recall = tp_c / n_gt

    # 101-point interpolation (COCO-style)
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 101):
        prec_at_r = precision[recall >= r_thresh]
        ap += (np.max(prec_at_r) if len(prec_at_r) > 0 else 0) / 101
    return float(ap)


def compute_recall_at_k(detections, ground_truth, k, iou_threshold=0.5):
    """Recall@k: fraction of GT found in top-k detections."""
    if not ground_truth:
        return 1.0
    if not detections:
        return 0.0
    dets = sorted(detections, key=lambda x: x["score"], reverse=True)[:k]
    found = sum(
        1 for gt in ground_truth
        if any(compute_iou(d["bbox"], gt["bbox"]) >= iou_threshold for d in dets)
    )
    return found / len(ground_truth)


def confidence_interval(data, confidence=0.95):
    """Compute mean and 95% CI."""
    from scipy import stats
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean), float(ci)


def cohens_d(group1, group2):
    """Effect size: Cohen's d."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0
