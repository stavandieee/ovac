import json, os, math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

# -----------------------------
# CONFIG: update these paths
# -----------------------------
THRESH = 0.3
IOU_THRESH = 0.5

# Default pseudo label paths (override via CLI if needed)
DEFAULT_PSEUDO = {
    "Swin-T only": "results/perception/visdrone/p1_grounding_dino_swint_visdrone.json",
    "Voted (no synonym)": "",
    "Voted + motor fix": "",
}

# -----------------------------
# Helper: class mapping
# -----------------------------
# VisDrone 10 classes
# ["pedestrian","person","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
def super_class(cls: str) -> str:
    c = cls.lower().strip()
    if c in {"pedestrian", "person"}:
        return "human"
    if c in {"bicycle", "motor", "tricycle", "awning-tricycle"}:
        return "cycle"
    if c in {"car", "van", "truck", "bus"}:
        return "vehicle"
    return "other"

# -----------------------------
# IoU
# -----------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# -----------------------------
# You already have these in your current script:
# - load_gt(...)
# - format of pseudo labels
# Reuse your existing load_gt() by copy/paste or import it.
# -----------------------------
def load_gt(labels_dir: str) -> Dict[str, List[dict]]:
    """Load VisDrone DET train GT from labels/*.txt.
    Returns dict[image_id] -> list of {"bbox":[x1,y1,x2,y2], "class":str}.
    VisDrone label format per line: x,y,w,h,score,object_category,truncation,occlusion
    """
    cls_id_to_name = {
        1:"pedestrian", 2:"person", 3:"bicycle", 4:"car", 5:"van",
        6:"truck", 7:"tricycle", 8:"awning-tricycle", 9:"bus", 10:"motor"
    }
    gt = defaultdict(list)
    labels_dir = os.path.expanduser(labels_dir)
    for fn in os.listdir(labels_dir):
        if not fn.endswith(".txt"):
            continue
        img_id = os.path.splitext(fn)[0]
        path = os.path.join(labels_dir, fn)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h = map(float, parts[:4])
                cls_id = int(float(parts[5]))
                cls = cls_id_to_name.get(cls_id)
                if cls is None:
                    continue
                bbox = [x, y, x + w, y + h]
                gt[img_id].append({"bbox": bbox, "class": cls})
    return gt

def normalize_pseudo(pseudo: dict) -> Dict[str, List[dict]]:
    """
    Normalize pseudo label json into dict[image_id] -> list of:
      {"bbox":[x1,y1,x2,y2], "class": str, "score": float}
    Adapt this to match your pseudo json schema.
    """
    # If your pseudo JSON already matches this, keep as-is.
    # Otherwise modify accordingly.
    out = defaultdict(list)
    for image_id, dets in pseudo.items():
        key = os.path.splitext(str(image_id))[0]
        for d in dets:
            cls = d.get("class") or d.get("label") or d.get("category")
            score = float(d.get("score", d.get("conf", 1.0)))
            bbox = d.get("bbox") or d.get("box")
            # support xywh
            if bbox and len(bbox) == 4 and d.get("bbox_format") == "xywh":
                x,y,w,h = bbox
                bbox = [x, y, x+w, y+h]
            out[key].append({"bbox": bbox, "class": str(cls), "score": score})
    return out

def match_and_score(pseudo_by_img: Dict[str, List[dict]], gt_by_img: Dict[str, List[dict]],
                    thresh: float, iou_thresh: float) -> dict:
    """
    Computes three metrics:
    - strict: class must match
    - objectness: ignore class
    - super: super-class must match

    Greedy one-to-one matching: each GT matched at most once.
    """
    # totals
    totals = {
        "strict": {"tp":0, "fp":0, "fn":0},
        "objectness": {"tp":0, "fp":0, "fn":0},
        "super": {"tp":0, "fp":0, "fn":0},
    }
    # per-class for strict (for your existing table)
    per_cls = defaultdict(lambda: {"tp":0, "fp":0, "fn":0})

    images_with_labels = 0
    total_pseudo = 0
    total_gt = sum(len(v) for v in gt_by_img.values())

    for img, gt_list in gt_by_img.items():
        dets = [d for d in pseudo_by_img.get(img, []) if d["score"] >= thresh and d["bbox"]]
        if dets:
            images_with_labels += 1
        total_pseudo += len(dets)

        # For objectness matching we only care about IoU
        gt_used_obj = [False]*len(gt_list)
        for d in dets:
            best_iou, best_j = 0.0, -1
            for j,g in enumerate(gt_list):
                if gt_used_obj[j]:
                    continue
                val = iou_xyxy(d["bbox"], g["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou >= iou_thresh and best_j >= 0:
                totals["objectness"]["tp"] += 1
                gt_used_obj[best_j] = True
            else:
                totals["objectness"]["fp"] += 1
        totals["objectness"]["fn"] += sum(1 for u in gt_used_obj if not u)

        # For strict and super we do separate matching, because class constraints change pairing
        # Strict
        gt_used = [False]*len(gt_list)
        for d in dets:
            best_iou, best_j = 0.0, -1
            for j,g in enumerate(gt_list):
                if gt_used[j]:
                    continue
                if d["class"] != g["class"]:
                    continue
                val = iou_xyxy(d["bbox"], g["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou >= iou_thresh and best_j >= 0:
                totals["strict"]["tp"] += 1
                per_cls[d["class"]]["tp"] += 1
                gt_used[best_j] = True
            else:
                totals["strict"]["fp"] += 1
                per_cls[d["class"]]["fp"] += 1
        # strict FN per class (count GT unmatched)
        for j,g in enumerate(gt_list):
            if not gt_used[j]:
                totals["strict"]["fn"] += 1
                per_cls[g["class"]]["fn"] += 1

        # Super-class
        gt_used_s = [False]*len(gt_list)
        for d in dets:
            sd = super_class(d["class"])
            best_iou, best_j = 0.0, -1
            for j,g in enumerate(gt_list):
                if gt_used_s[j]:
                    continue
                if sd != super_class(g["class"]):
                    continue
                val = iou_xyxy(d["bbox"], g["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou >= iou_thresh and best_j >= 0:
                totals["super"]["tp"] += 1
                gt_used_s[best_j] = True
            else:
                totals["super"]["fp"] += 1
        totals["super"]["fn"] += sum(1 for u in gt_used_s if not u)

    def prf(tp, fp, fn):
        p = tp / (tp+fp) if (tp+fp)>0 else 0.0
        r = tp / (tp+fn) if (tp+fn)>0 else 0.0
        return p, r

    out = {
        "threshold": thresh,
        "iou": iou_thresh,
        "total_gt": total_gt,
        "total_pseudo": total_pseudo,
        "images_with_labels": images_with_labels,
        "labels_per_image": (total_pseudo / images_with_labels) if images_with_labels>0 else 0.0,
        "strict": {},
        "objectness": {},
        "super": {},
        "per_class_strict": per_cls,
    }

    for k in ["strict","objectness","super"]:
        tp, fp, fn = totals[k]["tp"], totals[k]["fp"], totals[k]["fn"]
        p, r = prf(tp, fp, fn)
        out[k] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r,
            "noise_rate": (fp / total_pseudo) if total_pseudo>0 else 0.0,
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-labels", required=True, help="Path to VisDrone2019-DET-train/labels directory")
    ap.add_argument("--swint", default=DEFAULT_PSEUDO["Swin-T only"], help="Path to Swin-T pseudo label JSON")
    ap.add_argument("--voted", default=DEFAULT_PSEUDO["Voted (no synonym)"], help="Path to voted pseudo label JSON")
    ap.add_argument("--voted-motor", default=DEFAULT_PSEUDO["Voted + motor fix"], help="Path to voted+moter-fix pseudo label JSON")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--iou", type=float, default=IOU_THRESH)
    args = ap.parse_args()

    pseudo_files = {
        "Swin-T only": args.swint,
        "Voted (no synonym)": args.voted,
        "Voted + motor fix": args.voted_motor,
    }
    # Drop empty entries so you can run with only one file if needed
    pseudo_files = {k:v for k,v in pseudo_files.items() if v}

    print("Loading VisDrone train GT...")
    gt = load_gt(args.gt_labels)
    print(f"  {len(gt)} images, {sum(len(v) for v in gt.values())} boxes")

    all_results = {}
    print(f"\n{'='*78}")
    print(f"  TABLE 3 v2: Pseudo-Label Quality (IoU >= {args.iou}, τ >= {args.thresh})")
    print(f"{'='*78}")
    print(f"  {'Method':<22} {'Strict P':>9} {'Obj P':>9} {'Super P':>9} {'Labels/Img':>11} {'Total PL':>10}")
    print(f"  {'-'*78}")

    for name, path in pseudo_files.items():
        print(f"  Evaluating {name}... ", end="", flush=True)
        with open(path) as f:
            pseudo_raw = json.load(f)
        pseudo = normalize_pseudo(pseudo_raw)

        res = match_and_score(pseudo, gt, args.thresh, args.iou)
        all_results[name] = res

        sp = res["strict"]["precision"]
        op = res["objectness"]["precision"]
        sup = res["super"]["precision"]
        lpi = res["labels_per_image"]
        tpl = res["total_pseudo"]
        print(f"\r  {name:<22} {sp:>8.1%} {op:>8.1%} {sup:>8.1%} {lpi:>11.1f} {tpl:>10}")

    print(f"  {'-'*78}")

    os.makedirs("results/perception", exist_ok=True)
    out_path = "results/perception/table3_pseudo_label_quality_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ Saved to {out_path}")

if __name__ == "__main__":
    main()
