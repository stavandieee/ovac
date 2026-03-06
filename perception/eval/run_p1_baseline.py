"""
Experiment P1: Baseline Open-Vocabulary Detection on Aerial Data

Evaluates Grounding DINO and OWLv2 zero-shot on VisDrone/DOTA test sets.
No fine-tuning — just prompt with category names as text queries.

Usage:
    python eval_ovd_baseline.py --model grounding_dino_swint --dataset visdrone
    python eval_ovd_baseline.py --model owlv2_base --dataset visdrone
    python eval_ovd_baseline.py --all  # run all model-dataset combinations

Output:
    results/perception/p1_{model}_{dataset}.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str = "perception/configs/eval_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================
# Dataset Loaders
# =============================================================

class VisDroneLoader:
    """Load VisDrone-DET images and annotations."""

    CATEGORIES = {
        1: "pedestrian", 2: "person", 3: "car", 4: "van",
        5: "bus", 6: "truck", 7: "motor", 8: "bicycle",
        9: "awning-tricycle", 10: "tricycle"
    }

    def __init__(self, root: str, split: str = "test-dev"):
        split_map = {
            "train": "VisDrone2019-DET-train",
            "val": "VisDrone2019-DET-val",
            "test-dev": "VisDrone2019-DET-test-dev",
        }
        self.base = Path(root) / split_map[split]
        self.img_dir = self.base / "images"
        self.ann_dir = self.base / "annotations"
        self.image_files = sorted(self.img_dir.glob("*.jpg"))
        print(f"VisDrone {split}: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[Image.Image, List[dict]]:
        img_path = self.image_files[idx]
        ann_path = self.ann_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        annotations = []

        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 8:
                        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        score = int(parts[4])  # 0=ignored, 1=considered
                        cat_id = int(parts[5])
                        truncation = int(parts[6])
                        occlusion = int(parts[7])

                        if score == 0 or cat_id == 0 or cat_id == 11:
                            continue  # skip ignored regions and "others"

                        annotations.append({
                            "bbox": [x, y, x + w, y + h],  # xyxy format
                            "category_id": cat_id,
                            "category_name": self.CATEGORIES.get(cat_id, "unknown"),
                            "truncation": truncation,
                            "occlusion": occlusion,
                        })

        return image, annotations, str(img_path)


# =============================================================
# Model Wrappers
# =============================================================

class GroundingDINODetector:
    """Wrapper for Grounding DINO zero-shot detection."""

    def __init__(self, config_name: str, weights_path: str,
                 box_threshold: float = 0.25, text_threshold: float = 0.25):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        try:
            from groundingdino.util.inference import load_model, predict
            from groundingdino.util.inference import load_image as gd_load_image

            # Resolve config path
            import groundingdino
            gd_dir = Path(groundingdino.__file__).parent
            config_path = gd_dir / "config" / f"{config_name}.py"

            self.model = load_model(str(config_path), weights_path)
            self.predict_fn = predict
            self.load_image_fn = gd_load_image
            print(f"Loaded Grounding DINO: {config_name}")
        except ImportError:
            print("ERROR: groundingdino not installed.")
            print("Install: pip install groundingdino-py")
            raise

    def detect(self, image_path: str, text_queries: List[str]) -> List[dict]:
        """Run detection on a single image.

        Args:
            image_path: Path to image file
            text_queries: List of category names

        Returns:
            List of detections: [{bbox, score, category}]
        """
        image_source, image_tensor = self.load_image_fn(image_path)

        # Grounding DINO expects dot-separated prompt
        text_prompt = ". ".join(text_queries) + "."

        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        h, w = image_source.shape[:2]
        detections = []
        for box, score, phrase in zip(boxes, logits, phrases):
            # Convert from cxcywh normalized to xyxy pixel coords
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "category": phrase.strip().lower(),
            })

        return detections


class OWLv2Detector:
    """Wrapper for OWLv2 zero-shot detection."""

    def __init__(self, checkpoint: str, score_threshold: float = 0.1):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.score_threshold = score_threshold
        print(f"Loaded OWLv2: {checkpoint}")

    def detect(self, image_path: str, text_queries: List[str]) -> List[dict]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # OWLv2 expects list of lists of queries
        texts = [text_queries]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[h, w]])
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()

        results = self.processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "category": text_queries[int(label)].strip().lower(),
            })

        return detections


# =============================================================
# Metrics
# =============================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_ap(detections: List[dict], ground_truth: List[dict],
               iou_threshold: float = 0.5) -> float:
    """Compute Average Precision for a single category on a single image set."""
    if len(ground_truth) == 0:
        return 0.0 if len(detections) > 0 else 1.0
    if len(detections) == 0:
        return 0.0

    # Sort detections by score (descending)
    dets = sorted(detections, key=lambda x: x["score"], reverse=True)
    n_gt = len(ground_truth)
    matched = [False] * n_gt

    tp = []
    fp = []

    for det in dets:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            iou = compute_iou(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and not matched[best_gt_idx]:
            tp.append(1)
            fp.append(0)
            matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / n_gt

    # Compute AP using 11-point interpolation
    ap = 0.0
    for r_thresh in np.arange(0, 1.1, 0.1):
        prec_at_recall = precision[recall >= r_thresh]
        if len(prec_at_recall) > 0:
            ap += np.max(prec_at_recall) / 11.0

    return ap


def compute_recall_at_k(detections: List[dict], ground_truth: List[dict],
                        k: int, iou_threshold: float = 0.5) -> float:
    """Compute Recall@k: fraction of GT objects found in top-k detections."""
    if len(ground_truth) == 0:
        return 1.0
    if len(detections) == 0:
        return 0.0

    dets = sorted(detections, key=lambda x: x["score"], reverse=True)[:k]
    matched = 0

    for gt in ground_truth:
        for det in dets:
            if compute_iou(det["bbox"], gt["bbox"]) >= iou_threshold:
                matched += 1
                break

    return matched / len(ground_truth)


# =============================================================
# Main Evaluation Loop
# =============================================================

def evaluate_model(
    detector,
    dataset,
    text_queries: List[str],
    novel_categories: List[str],
    results_path: str,
    max_images: int = None,
):
    """Run full evaluation and save results."""

    all_detections = {}  # category -> list of (score, is_tp) across all images
    all_gt_counts = {}
    recall_at_k_accum = {k: [] for k in [1, 5, 10]}

    n_images = min(len(dataset), max_images) if max_images else len(dataset)

    for idx in tqdm(range(n_images), desc="Evaluating"):
        image, gt_anns, img_path = dataset[idx]

        # Run detection
        detections = detector.detect(img_path, text_queries)

        # Match detections to text queries -> category mapping
        for cat_name in text_queries:
            cat_dets = [d for d in detections if cat_name.lower() in d["category"].lower()]
            cat_gts = [g for g in gt_anns if g["category_name"].lower() == cat_name.lower()]

            if cat_name not in all_detections:
                all_detections[cat_name] = []
                all_gt_counts[cat_name] = 0

            all_gt_counts[cat_name] += len(cat_gts)

            # Per-image AP contribution
            if len(cat_gts) > 0 or len(cat_dets) > 0:
                ap = compute_ap(cat_dets, cat_gts, iou_threshold=0.5)
                all_detections[cat_name].append(ap)

        # Recall@k for novel categories
        novel_dets = [d for d in detections
                      if any(nc.lower() in d["category"].lower() for nc in novel_categories)]
        novel_gts = [g for g in gt_anns if g["category_name"].lower() in
                     [nc.lower() for nc in novel_categories]]

        for k in [1, 5, 10]:
            r_at_k = compute_recall_at_k(novel_dets, novel_gts, k)
            recall_at_k_accum[k].append(r_at_k)

    # Compute aggregate metrics
    results = {
        "model": detector.__class__.__name__,
        "n_images": n_images,
        "per_category_ap": {},
        "seen_mAP50": 0.0,
        "novel_mAP50": 0.0,
        "overall_mAP50": 0.0,
        "recall_at_k": {},
    }

    seen_aps = []
    novel_aps = []

    for cat_name, ap_list in all_detections.items():
        if len(ap_list) > 0:
            mean_ap = np.mean(ap_list)
        else:
            mean_ap = 0.0

        results["per_category_ap"][cat_name] = float(mean_ap)

        if cat_name.lower() in [nc.lower() for nc in novel_categories]:
            novel_aps.append(mean_ap)
        else:
            seen_aps.append(mean_ap)

    results["seen_mAP50"] = float(np.mean(seen_aps)) if seen_aps else 0.0
    results["novel_mAP50"] = float(np.mean(novel_aps)) if novel_aps else 0.0
    results["overall_mAP50"] = float(np.mean(seen_aps + novel_aps)) if (seen_aps + novel_aps) else 0.0

    for k in [1, 5, 10]:
        vals = recall_at_k_accum[k]
        results["recall_at_k"][f"R@{k}"] = float(np.mean(vals)) if vals else 0.0

    results["gt_counts"] = {k: v for k, v in all_gt_counts.items()}

    # Save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {results_path}")
    print(f"  Seen mAP@50:    {results['seen_mAP50']:.4f}")
    print(f"  Novel mAP@50:   {results['novel_mAP50']:.4f}")
    print(f"  Overall mAP@50: {results['overall_mAP50']:.4f}")
    for k_str, v in results["recall_at_k"].items():
        print(f"  {k_str} (novel):  {v:.4f}")
    print(f"{'='*60}\n")

    return results


def build_detector(model_name: str, config: dict):
    """Factory function to build detector from config."""
    model_cfg = config["models"][model_name]
    weights_dir = config["data"]["weights_dir"]

    if "grounding_dino" in model_name:
        return GroundingDINODetector(
            config_name=model_cfg["config"],
            weights_path=os.path.join(weights_dir, model_cfg["weights"]),
            box_threshold=model_cfg["box_threshold"],
            text_threshold=model_cfg["text_threshold"],
        )
    elif "owlv2" in model_name:
        return OWLv2Detector(
            checkpoint=model_cfg["checkpoint"],
            score_threshold=model_cfg["score_threshold"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="P1: Baseline OVD Evaluation")
    parser.add_argument("--model", type=str, default="grounding_dino_swint",
                        choices=["grounding_dino_swint", "grounding_dino_swinb",
                                 "owlv2_base", "owlv2_large"])
    parser.add_argument("--dataset", type=str, default="visdrone",
                        choices=["visdrone", "dota"])
    parser.add_argument("--config", type=str, default="perception/configs/eval_config.yaml")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images (for debugging)")
    parser.add_argument("--all", action="store_true",
                        help="Run all model-dataset combinations")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config["data"]["results_dir"]

    if args.all:
        models = config["exp_p1"]["models"]
        datasets = config["exp_p1"]["datasets"]
    else:
        models = [args.model]
        datasets = [args.dataset]

    for model_name in models:
        detector = build_detector(model_name, config)

        for dataset_name in datasets:
            print(f"\n{'#'*60}")
            print(f"Evaluating {model_name} on {dataset_name}")
            print(f"{'#'*60}")

            # Load dataset
            if dataset_name == "visdrone":
                dataset = VisDroneLoader(
                    root=config["data"]["visdrone_root"],
                    split="test-dev"
                )
                text_queries = config["visdrone"]["all_categories"]
                novel_cats = config["visdrone"]["novel_categories"]
            else:
                # DOTA loader would go here
                print(f"DOTA loader not yet implemented — skipping")
                continue

            results_path = os.path.join(results_dir, f"p1_{model_name}_{dataset_name}.json")

            evaluate_model(
                detector=detector,
                dataset=dataset,
                text_queries=text_queries,
                novel_categories=novel_cats,
                results_path=results_path,
                max_images=args.max_images,
            )


if __name__ == "__main__":
    main()
