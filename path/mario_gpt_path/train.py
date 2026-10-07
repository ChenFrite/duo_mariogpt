import os
import sys
import torch

from mario_gpt_path import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mario_gpt_path.level import FULL_LEVEL_STR_WITH_PATHS

# ==========================================
# 設定區
# ==========================================
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"

# 從隨機初始化開始訓練，或改成 checkpoint 路徑繼續訓練
# 例如: "./Mario-GPT2-700-context-length_29/iteration_9000"
BASE = "random"

OUTPUT_DIR = "Mario-GPT2-700-context-length_29"

# ==========================================
# 主程式
# ==========================================
mario_lm = MarioLM(lm_path=BASE, tokenizer_path=TOKENIZER_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
mario_lm = mario_lm.to(device)

dataset = MarioDataset(mario_lm.tokenizer, level_string=FULL_LEVEL_STR_WITH_PATHS.strip())

config = TrainingConfig(
    mixed_precision="bf16",
    output_dir=OUTPUT_DIR,
    learning_rate=5e-4,
    batch_size=4,
    save_iteration=1000,
    eval_iteration=1000,
    total_steps=1001,
)

trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

print("Start training 2dRelPos MarioGPT...")
trainer.train()