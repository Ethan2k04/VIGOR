#!/bin/bash
# =============================================================================
# setup.sh - 一键下载数据集、解压、下载模型权重
# 使用方法: bash setup.sh
# 请将此脚本放在 model_training/ 目录下运行
# =============================================================================

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " 当前工作目录: $SCRIPT_DIR"
echo "=================================================="

REPO_ID="Ethan2k04/GB3DV-25k"
DATASET_CACHE="./ms_dataset_cache"
INPUT_LATENT_DIR="./input_latent"

# --------------------------------------------------------------------------
# Step 1: 检查依赖
# --------------------------------------------------------------------------
echo ""
echo "[Step 1/4] 检查依赖..."

if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3，请先安装。"
    exit 1
fi

python3 -c "import modelscope" 2>/dev/null || {
    echo "未找到 modelscope，正在安装..."
    pip install modelscope -q
}

echo "依赖检查完成。"

# --------------------------------------------------------------------------
# Step 2: 从 ModelScope 下载数据集文件（16 个 part）
# --------------------------------------------------------------------------
echo ""
echo "[Step 2/4] 从 ModelScope 下载数据集..."
echo "  仓库: $REPO_ID"

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
        print(f"  已存在，跳过: {filename}")
        continue
    print(f"  正在下载: {filename} ...")
    try:
        downloaded_path = dataset_file_download(
            dataset_id=REPO_ID,
            file_path=filename,
            cache_dir=CACHE_DIR,
        )
        # 移动到期望位置（dataset_file_download 可能放在子目录）
        if downloaded_path != dest:
            import shutil
            shutil.move(downloaded_path, dest)
        print(f"  ✓ 下载完成: {filename}")
    except Exception as e:
        print(f"  ✗ 下载失败: {filename} -> {e}")
        raise
PYEOF

echo "数据集文件下载完成。"

# --------------------------------------------------------------------------
# Step 3: 解压全部 16 个 tar 到 input_latent/
# --------------------------------------------------------------------------
echo ""
echo "[Step 3/4] 解压 tar 文件..."

mkdir -p "$INPUT_LATENT_DIR"

for i in $(seq 1 16); do
    TAR_FILE="input_latent_part${i}.tar"
    TAR_PATH="$DATASET_CACHE/$TAR_FILE"

    if [ ! -f "$TAR_PATH" ]; then
        echo "  ✗ 找不到文件: $TAR_PATH，跳过。"
        continue
    fi

    echo "  正在解压: $TAR_FILE -> input_latent/ ..."
    tar -xf "$TAR_PATH" -C "$(dirname "$INPUT_LATENT_DIR")"
    echo "  ✓ 解压完成: $TAR_FILE"
done

# 复制 json 到 model_training 根目录
if [ -f "$DATASET_CACHE/annotated_metadata.json" ]; then
    cp "$DATASET_CACHE/annotated_metadata.json" ./annotated_metadata.json
    echo "  ✓ annotated_metadata.json 已复制到当前目录"
fi

echo "解压完成。"

# --------------------------------------------------------------------------
# Step 4: 下载 Wan2.1-T2V-1.3B 模型权重
# --------------------------------------------------------------------------
echo ""
echo "[Step 4/4] 下载 Wan2.1-T2V-1.3B 模型权重..."

if [ -d "./Wan2.1-T2V-1.3B" ] && [ "$(ls -A ./Wan2.1-T2V-1.3B)" ]; then
    echo "  Wan2.1-T2V-1.3B 目录已存在且非空，跳过下载。"
else
    modelscope download Wan-AI/Wan2.1-T2V-1.3B --local_dir ./Wan2.1-T2V-1.3B
    echo "  ✓ 模型权重下载完成"
fi

# --------------------------------------------------------------------------
# 完成，打印最终目录结构
# --------------------------------------------------------------------------
echo ""
echo "=================================================="
echo " 全部完成！最终目录结构："
echo "=================================================="
echo "model_training/"
echo "├── setup.sh"
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