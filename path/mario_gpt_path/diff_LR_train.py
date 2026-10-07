import os
import sys
import torch

from mario_gpt_path import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt_path.level import FULL_LEVEL_STR_WITH_PATHS

# ================= 設定區 =================
TOKENIZER_PATH  = "shyamsn97/Mario-GPT2-700-context-length"

# 輸出目錄前綴，每輪會加上 _round{N}
OUTPUT_PREFIX   = "M_multiround"

# 重複執行次數
NUM_ROUNDS      = 10

# 每輪訓練步數
STEPS_PER_ROUND = 10001

# LR 設定：第一輪 / 後續輪
LR_FIRST        = 5e-4
LR_CONTINUE     = 1e-4

# 其他超參數
BATCH_SIZE      = 4
LR_WARMUP_STEPS = 150
SAVE_ITERATION  = 10000
EVAL_ITERATION  = 1000
MIXED_PRECISION = "bf16"
# ==========================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for round_idx in range(1, NUM_ROUNDS + 1):
    print(f"\n{'='*60}")
    print(f"  ROUND {round_idx} / {NUM_ROUNDS}")
    print(f"{'='*60}")

    if round_idx == 1:
        model_path = "random"
        lr         = LR_FIRST
    else:
        prev_output = os.path.join(THIS_DIR, f"{OUTPUT_PREFIX}_round{round_idx - 1}")
        ckpt_dirs = sorted(
            [d for d in os.listdir(prev_output) if d.startswith("iteration_")],
            key=lambda d: int(d.split("_")[1])
        )
        assert ckpt_dirs, f"找不到上一輪 checkpoint：{prev_output}"
        model_path = os.path.join(prev_output, ckpt_dirs[-1])
        lr         = LR_CONTINUE

    output_dir = os.path.join(THIS_DIR, f"{OUTPUT_PREFIX}_round{round_idx}")

    print(f"  輸入模型：{model_path}")
    print(f"  輸出目錄：{output_dir}")
    print(f"  LR={lr}, steps={STEPS_PER_ROUND}")

    mario_lm = MarioLM(lm_path=model_path, tokenizer_path=TOKENIZER_PATH)
    mario_lm = mario_lm.to(device)

    dataset = MarioDataset(mario_lm.tokenizer, level_string=FULL_LEVEL_STR_WITH_PATHS.strip())

    config = TrainingConfig(
        mixed_precision        = MIXED_PRECISION,
        output_dir             = output_dir,
        learning_rate          = lr,
        lr_warmup_steps        = LR_WARMUP_STEPS,
        batch_size             = BATCH_SIZE,
        save_iteration         = SAVE_ITERATION,
        eval_iteration         = EVAL_ITERATION,
        total_steps            = STEPS_PER_ROUND,
    )

    trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

    print(f"  開始訓練 round {round_idx}...")
    trainer.train()
    print(f"  Round {round_idx} 完成，模型已存至 {output_dir}")

    del trainer, mario_lm, dataset
    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"  全部 {NUM_ROUNDS} 輪訓練完成！")
print(f"{'='*60}")
