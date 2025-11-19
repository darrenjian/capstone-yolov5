#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --output=process_tears_%j.out
#SBATCH --job-name=process_tears

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=================================================="

# Go to repo root
cd /gpfs/home/pb3060/capstone-yolov5

# Create or activate virtualenv
if [ -d "venv" ]; then
    echo "Activating existing virtual environment..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    # echo "Installing required packages..."
    # pip install -r requirements.txt
fi

# Print Python and package versions
echo ""
echo "Python version:"
python --version
echo ""
echo "Installed packages:"
pip list | grep -E "(pandas|numpy|pydicom|tqdm|pyyaml)"
echo ""

# Run the batch conversion
echo "=================================================="
echo "Starting meniscus tear dataset processing..."
echo "=================================================="

python capstone_scripts/batch_convert_annotations.py \
    --data_source meniscus_tear \
    --output yolo_dataset_tears \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Dataset processing completed successfully!"
    echo "=================================================="
    echo "Output location: yolo_dataset_tears/"
    echo ""
    echo "Dataset statistics:"
    if [ -f "yolo_dataset_tears/conversion_report.json" ]; then
        python3 -c "
import json
with open('yolo_dataset_tears/conversion_report.json') as f:
    data = json.load(f)
    summary = data['summary']
    print(f\"  Total volumes: {summary['total_volumes']}\")
    print(f\"  Total images: {summary['total_images']}\")
    print(f\"  Total annotations: {summary['total_annotations']}\")
    print(f\"  Train volumes: {summary['train_volumes']}\")
    print(f\"  Val volumes: {summary['val_volumes']}\")
    print(f\"  Test volumes: {summary['test_volumes']}\")
"
    fi
else
    echo ""
    echo "=================================================="
    echo "❌ Dataset processing failed!"
    echo "=================================================="
    exit 1
fi

# Print completion time
echo ""
echo "End Time: $(date)"
echo "=================================================="