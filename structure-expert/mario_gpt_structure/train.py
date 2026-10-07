import os
import sys
import torch

from mario_gpt_structure import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt_structure.dataset import PATCH_WIDTH, FUTURE_WIDTH

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
BASE           = "random"
# 新目錄：舊的 Mario-GPT2-700-context-length_29 是 path-only 模型，不可混用
OUTPUT_DIR     = os.path.join(_THIS_DIR, "Structure-GPT2-v2")
VAL_FRACTION   = 0.1   # 關卡最後 10% 的 patch 當 held-out val

if __name__ == "__main__":
    if os.path.isdir(OUTPUT_DIR) and any(d.startswith("iteration_") for d in os.listdir(OUTPUT_DIR)):
        sys.exit(f"{OUTPUT_DIR} 已有 checkpoint；請換 OUTPUT_DIR 或先移走，避免新舊 log/checkpoint 混在一起。")

    mario_lm = MarioLM(lm_path=BASE, tokenizer_path=TOKENIZER_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mario_lm = mario_lm.to(device)

    dataset = MarioDataset(mario_lm.tokenizer)

    # stride=1 的 patch 彼此重疊 PATCH_WIDTH+FUTURE_WIDTH 欄；在 train/val 之間留同寬的間隔，避免 val 洩漏
    n = len(dataset)
    val_start = int(n * (1 - VAL_FRACTION))
    gap = PATCH_WIDTH + FUTURE_WIDTH
    train_indices = list(range(0, val_start - gap))
    val_indices   = list(range(val_start, n))
    print(f"Split: train={len(train_indices)}  val={len(val_indices)}  gap={gap}")

    config = TrainingConfig(
        mixed_precision="bf16",
        output_dir=OUTPUT_DIR,
        learning_rate=5e-4,
        batch_size=4,
        save_iteration=1000,
        eval_iteration=1000,
        total_steps=10001,
    )

    trainer = MarioGPTTrainer(
        mario_lm, dataset, config=config,
        train_indices=train_indices, val_indices=val_indices,
    )

    print("Start training structure expert ([path | future path | structure] format)...")
    trainer.train()
