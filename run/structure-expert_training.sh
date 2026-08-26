#!/bin/bash
# 執行 structure-expert 從頭訓練

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo " Structure Expert Training"
echo "======================================"
cd "$SCRIPT_DIR/../structure" || exit 1
pip install -e .
if [ $? -ne 0 ]; then
    echo "[ERROR] pip install -e . failed. Aborting."
    exit 1
fi
cd mario_gpt || exit 1
python diff_LR_train.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Training failed."
    exit 1
fi

echo ""
echo "======================================"
echo " Training done!"
echo "======================================"
