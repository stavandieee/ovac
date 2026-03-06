#!/bin/bash
# Download all required datasets for the aerial perception experiments
# Run once before starting experiments

set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "Downloading datasets for aerial OVD research"
echo "Target directory: $DATA_DIR"
echo "============================================"

# ---------------------------------------------------
# 1. VisDrone-DET 2021
# ---------------------------------------------------
echo ""
echo "[1/3] VisDrone-DET 2021"
echo "  Source: https://github.com/VisDrone/VisDrone-Dataset"
echo "  10,209 images, 10 categories"
echo ""

VISDRONE_DIR="$DATA_DIR/visdrone"
mkdir -p "$VISDRONE_DIR"

if [ ! -d "$VISDRONE_DIR/VisDrone2019-DET-train" ]; then
    echo "  Downloading VisDrone train split..."
    wget -q --show-progress -O "$VISDRONE_DIR/train.zip" \
        "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip"
    unzip -q "$VISDRONE_DIR/train.zip" -d "$VISDRONE_DIR"
    rm "$VISDRONE_DIR/train.zip"
    echo "  ✓ Train split ready"
else
    echo "  ✓ Train split already exists"
fi

if [ ! -d "$VISDRONE_DIR/VisDrone2019-DET-val" ]; then
    echo "  Downloading VisDrone val split..."
    wget -q --show-progress -O "$VISDRONE_DIR/val.zip" \
        "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-val.zip"
    unzip -q "$VISDRONE_DIR/val.zip" -d "$VISDRONE_DIR"
    rm "$VISDRONE_DIR/val.zip"
    echo "  ✓ Val split ready"
else
    echo "  ✓ Val split already exists"
fi

if [ ! -d "$VISDRONE_DIR/VisDrone2019-DET-test-dev" ]; then
    echo "  Downloading VisDrone test-dev split..."
    wget -q --show-progress -O "$VISDRONE_DIR/test-dev.zip" \
        "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"
    unzip -q "$VISDRONE_DIR/test-dev.zip" -d "$VISDRONE_DIR"
    rm "$VISDRONE_DIR/test-dev.zip"
    echo "  ✓ Test-dev split ready"
else
    echo "  ✓ Test-dev split already exists"
fi

# ---------------------------------------------------
# 2. Grounding DINO Weights
# ---------------------------------------------------
echo ""
echo "[2/3] Model Checkpoints"
echo ""

WEIGHTS_DIR="$DATA_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

if [ ! -f "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "  Downloading Grounding DINO (Swin-T)..."
    wget -q --show-progress -O "$WEIGHTS_DIR/groundingdino_swint_ogc.pth" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    echo "  ✓ Grounding DINO Swin-T ready"
else
    echo "  ✓ Grounding DINO Swin-T already exists"
fi

if [ ! -f "$WEIGHTS_DIR/groundingdino_swinb_cogcoor.pth" ]; then
    echo "  Downloading Grounding DINO (Swin-B)..."
    wget -q --show-progress -O "$WEIGHTS_DIR/groundingdino_swinb_cogcoor.pth" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    echo "  ✓ Grounding DINO Swin-B ready"
else
    echo "  ✓ Grounding DINO Swin-B already exists"
fi

# OWLv2 and CLIP are downloaded automatically via HuggingFace transformers
echo "  ℹ OWLv2 and CLIP weights will be auto-downloaded on first use via HuggingFace"

# ---------------------------------------------------
# 3. DOTA v2.0 (manual download required)
# ---------------------------------------------------
echo ""
echo "[3/3] DOTA v2.0"
echo "  ⚠ DOTA v2.0 requires manual download from: https://captain-whu.github.io/DOTA/"
echo "  ⚠ You need to register and request access"
echo "  ⚠ Once downloaded, place in: $DATA_DIR/dota/"
echo "  ⚠ Expected structure:"
echo "      $DATA_DIR/dota/train/images/"
echo "      $DATA_DIR/dota/train/labelTxt-v2.0/"
echo "      $DATA_DIR/dota/val/images/"
echo "      $DATA_DIR/dota/val/labelTxt-v2.0/"
echo ""

# ---------------------------------------------------
# Summary
# ---------------------------------------------------
echo "============================================"
echo "Download Summary:"
echo "  ✓ VisDrone-DET 2021 (train/val/test-dev)"
echo "  ✓ Grounding DINO (Swin-T, Swin-B)"
echo "  ℹ OWLv2, CLIP: auto-download via HuggingFace"
echo "  ⚠ DOTA v2.0: manual download required"
echo ""
echo "Total disk usage:"
du -sh "$DATA_DIR" 2>/dev/null || echo "  (run after download completes)"
echo "============================================"
