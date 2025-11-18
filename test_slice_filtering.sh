#!/bin/bash
# Quick test script for slice filtering feature
# This script runs through all steps to verify the implementation

set -e  # Exit on error

# Configuration
DATASET_ROOT="${1:-/gpfs/data/lattanzilab/Ilias/yolo_dataset_meniscus}"
OUTPUT_DIR="test_slice_filtering_output"

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Output:  $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$(dirname "$0")/.."  # Go to project root

# Step 1: Calculate slice range
python capstone_scripts/calculate_slice_range.py \
    --dataset_root "$DATASET_ROOT" \
    --output "$OUTPUT_DIR/slice_range.json" \
    --splits train

if [ $? -eq 0 ]; then
    echo "✓ Slice range calculation completed successfully"
else
    echo "✗ Slice range calculation failed"
    exit 1
fi

# Step 2: Validate filtering
python capstone_scripts/validate_slice_filtering.py \
    --dataset_root "$DATASET_ROOT" \
    --slice_range "$OUTPUT_DIR/slice_range.json" \
    --split train

if [ $? -eq 0 ]; then
    echo "✓ Slice filtering validation passed"
else
    echo "✗ Slice filtering validation failed"
    exit 1
fi

# Step 3: Show statistics
echo "Slice range statistics saved to:"
echo "  $OUTPUT_DIR/slice_range.json"
echo ""
echo "To view the statistics:"
echo "  cat $OUTPUT_DIR/slice_range.json | python -m json.tool"
echo ""
echo "To use in training, add this flag:"
echo "  --slice-range $OUTPUT_DIR/slice_range.json"
echo ""
echo "Example training command:"
echo "  python train.py \\"
echo "    --data data/meniscus.yaml \\"
echo "    --weights yolov5s.pt \\"
echo "    --epochs 100 \\"
echo "    --slice-range $OUTPUT_DIR/slice_range.json"