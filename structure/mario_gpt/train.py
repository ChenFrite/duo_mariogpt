import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
BASE           = "random"
OUTPUT_DIR     = "Mario-GPT2-700-context-length_29"

mario_lm = MarioLM(lm_path=BASE, tokenizer_path=TOKENIZER_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
mario_lm = mario_lm.to(device)

dataset = MarioDataset(mario_lm.tokenizer)

config = TrainingConfig(
    mixed_precision="bf16",
    output_dir=OUTPUT_DIR,
    learning_rate=5e-4,
    batch_size=4,
    save_iteration=1000,
    eval_iteration=10000,
    total_steps=10001,
)

trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

print("Start training 4-column path-prefix MarioGPT...")
trainer.train()
