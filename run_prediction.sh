#!/bin/bash

# ESM Structure Prediction Script
# Usage: ./run_prediction.sh <input_json_path> <output_dir> <model_dir>

set -e

# Check if correct number of arguments provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_json_path> <output_dir> <model_dir>"
    echo ""
    echo "Arguments:"
    echo "  input_json_path  Path to input JSON file (can be absolute or relative)"
    echo "  output_dir       Directory where results will be saved"
    echo "  model_dir        Directory where model files are stored/downloaded"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/input.json ./results /data/protein/torch_model/hub"
    exit 1
fi

INPUT_JSON="$1"
OUTPUT_DIR="$2"
MODEL_DIR="$3"

# Convert to absolute paths
INPUT_JSON=$(realpath "$INPUT_JSON")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
MODEL_DIR=$(realpath "$MODEL_DIR")

# Get the directory containing this script (ESM project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Validate input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file '$INPUT_JSON' does not exist!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Get the parent directory of the input file for mounting
INPUT_DIR=$(dirname "$INPUT_JSON")
INPUT_FILENAME=$(basename "$INPUT_JSON")

# Get the parent directory of the output for mounting
OUTPUT_PARENT_DIR=$(dirname "$OUTPUT_DIR")
OUTPUT_DIRNAME=$(basename "$OUTPUT_DIR")

# Get the parent directory of the model dir for mounting
MODEL_PARENT_DIR=$(dirname "$MODEL_DIR")

# Get current user and group IDs to maintain file ownership
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

echo "============================================"
echo "ESM Structure Prediction"
echo "============================================"
echo "Input file:    $INPUT_JSON"
echo "Output dir:    $OUTPUT_DIR"
echo "Model dir:     $MODEL_DIR"
echo "Running as:    $USER_NAME ($USER_ID:$GROUP_ID)"
echo "============================================"
echo ""

# Run Docker container with appropriate volume mounts and user permissions
docker run --gpus all \
    --user "$USER_ID:$GROUP_ID" \
    -v "$SCRIPT_DIR":/workspace \
    -v "$INPUT_DIR":/input_mount \
    -v "$OUTPUT_PARENT_DIR":/output_mount \
    -v "$MODEL_PARENT_DIR":/model_mount \
    -e HOME=/tmp \
    -w /workspace \
    esm-predict \
    bash -c "
        # Create temporary user if needed for proper file permissions
        if ! id -u $USER_ID >/dev/null 2>&1; then
            echo 'Creating temporary user for proper file permissions...'
            groupadd -g $GROUP_ID tempgroup 2>/dev/null || true
            useradd -u $USER_ID -g $GROUP_ID -d /tmp -s /bin/bash tempuser 2>/dev/null || true
        fi
        
        # Run the prediction script
        python scripts/predict_structure.py \
            -i \"/input_mount/$INPUT_FILENAME\" \
            -o \"/output_mount/$OUTPUT_DIRNAME\" \
            -m \"/model_mount/$(basename "$MODEL_DIR")\"
    "

echo ""
echo "============================================"
echo "Prediction completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
