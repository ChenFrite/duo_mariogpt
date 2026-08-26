#!/bin/bash
# 用法: bash run_pipeline.sh [COUNT]
# COUNT: 生成次數，預設 10

COUNT=${1:-10}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo " STEP 1: Path Expert (count=$COUNT)"
echo "======================================"
cd "$SCRIPT_DIR/../path" || exit 1
pip install -e .
if [ $? -ne 0 ]; then
    echo "[ERROR] pip install -e . failed (path). Aborting."
    exit 1
fi
cd mario_gpt || exit 1
python multi_test.py --count "$COUNT"
if [ $? -ne 0 ]; then
    echo "[ERROR] Path expert failed. Aborting."
    exit 1
fi

echo ""
echo "======================================"
echo " STEP 2: Structure Expert"
echo "======================================"
cd "$SCRIPT_DIR/../structure" || exit 1
pip install -e .
if [ $? -ne 0 ]; then
    echo "[ERROR] pip install -e . failed (structure). Aborting."
    exit 1
fi
cd mario_gpt || exit 1
python pipeline_pathexpert.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Structure expert failed."
    exit 1
fi

echo ""
echo "======================================"
echo " Pipeline done!"
echo "======================================"
