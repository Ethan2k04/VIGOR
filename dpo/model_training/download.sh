#!/bin/bash
# =============================================================================
# download.sh - download the dataset, extract it, and download the model weights
# Usage: bash download.sh
# Run this script from the model_training/ directory
# =============================================================================

set -e  # exit immediately on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " Current working directory: $SCRIPT_DIR"
echo "=================================================="

REPO_ID="Ethan2k04/GB3DV-25k"
DATASET_CACHE="./ms_dataset_cache"
INPUT_LATENT_DIR="./input_latent"

# --------------------------------------------------------------------------
# Step 1: check dependencies
# --------------------------------------------------------------------------
echo ""
echo "[Step 1/4] Checking dependencies..."

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found, please install it first."
    exit 1
fi

python3 -c "import modelscope" 2>/dev/null || {
    echo "modelscope not found, installing..."
    pip install modelscope -q
}

echo "Dependency check complete."

# --------------------------------------------------------------------------
# Step 2: download the dataset files from ModelScope (16 parts)
# --------------------------------------------------------------------------
echo ""
echo "[Step 2/4] Downloading dataset from ModelScope..."
echo "  Repository: $REPO_ID"

mkdir -p "$DATASET_CACHE"

python3 - <<PYEOF
from modelscope.hub.file_download import dataset_file_download
import os

REPO_ID = "Ethan2k04/GB3DV-25k"
CACHE_DIR = "$DATASET_CACHE"

files_to_download = [
    *[f"input_latent_part{i}.tar" for i in range(1, 17)],
    "annotated_metadata.json",
]

for filename in files_to_download:
    dest = os.path.join(CACHE_DIR, filename)
    if os.path.exists(dest):
        print(f"  Already exists, skipping: {filename}")
        continue
    print(f"  Downloading: {filename} ...")
    try:
        downloaded_path = dataset_file_download(
            dataset_id=REPO_ID,
            file_path=filename,
            cache_dir=CACHE_DIR,
        )
        # Move to the expected location (dataset_file_download may place it in a subdirectory)
        if downloaded_path != dest:
            import shutil
            shutil.move(downloaded_path, dest)
        print(f"  ✓ Download complete: {filename}")
    except Exception as e:
        print(f"  ✗ Download failed: {filename} -> {e}")
        raise
PYEOF

echo "Dataset files downloaded."

# --------------------------------------------------------------------------
# Step 3: extract all 16 tar files into input_latent/
# --------------------------------------------------------------------------
echo ""
echo "[Step 3/4] Extracting tar files..."

mkdir -p "$INPUT_LATENT_DIR"

for i in $(seq 1 16); do
    TAR_FILE="input_latent_part${i}.tar"
    TAR_PATH="$DATASET_CACHE/$TAR_FILE"

    if [ ! -f "$TAR_PATH" ]; then
        echo "  ✗ File not found: $TAR_PATH, skipping."
        continue
    fi

    echo "  Extracting: $TAR_FILE -> input_latent/ ..."
    tar -xf "$TAR_PATH" -C "$(dirname "$INPUT_LATENT_DIR")"
    echo "  ✓ Extracted: $TAR_FILE"
done

# Copy the json to the model_training root directory
if [ -f "$DATASET_CACHE/annotated_metadata.json" ]; then
    cp "$DATASET_CACHE/annotated_metadata.json" ./annotated_metadata.json
    echo "  ✓ annotated_metadata.json copied to the current directory"
fi

echo "Extraction complete."

# --------------------------------------------------------------------------
# Step 4: download the Wan2.1-T2V-1.3B model weights
# --------------------------------------------------------------------------
echo ""
echo "[Step 4/4] Downloading Wan2.1-T2V-1.3B model weights..."

if [ -d "./Wan2.1-T2V-1.3B" ] && [ "$(ls -A ./Wan2.1-T2V-1.3B)" ]; then
    echo "  Wan2.1-T2V-1.3B directory already exists and is non-empty, skipping download."
else
    modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir ./Wan2.1-T2V-1.3B
    echo "  ✓ Model weights downloaded"
fi

# --------------------------------------------------------------------------
# Done — print the final directory structure
# --------------------------------------------------------------------------
echo ""
echo "=================================================="
echo " All done! Final directory structure:"
echo "=================================================="
echo "model_training/"
echo "├── download.sh"
echo "├── annotated_metadata.json"
echo "├── input_latent/"
echo "│   ├── static_indoor/"
echo "│   ├── static_outdoor/"
echo "│   ├── static_dynamic_indoor/"
echo "│   └── static_dynamic_outdoor/"
echo "└── Wan2.1-T2V-1.3B/"
echo ""

if command -v tree &>/dev/null; then
    tree -L 2 .
fi