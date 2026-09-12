import torch
import os
import sys
import random
import argparse
import numpy as np
from typing import Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mario_gpt import MarioLM
from mario_gpt.dataset import build_positions_2d

# ==========================================
# 1. 全域設定區
# ==========================================
GENERATE_COUNT = 10

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
CHECKPOINT_DIR = "./M_multiround_round1/iteration_10000"

OUTPUT_DIR = "multi_test"

QUANTIFIERS = ["no", "little", "some", "many"]

# 與 prompter.py 相同的 threshold
STATISTICS = {
    "M": np.array([0.0, 10.0, 30.0]),
    "J": np.array([0.0, 5.0, 15.0]),
    "j": np.array([0.0, 5.0, 15.0]),
    "A": np.array([0.0, 5.0, 15.0]),
    "C": np.array([0.0, 2.0, 8.0]),
}

GLOBAL_STATS = {
    "M": {"correct": 0, "total": 0},
    "J": {"correct": 0, "total": 0},
    "j": {"correct": 0, "total": 0},
    "A": {"correct": 0, "total": 0},
    "C": {"correct": 0, "total": 0},
}

# ==========================================
# 2. PromptJudge
# ==========================================
class PromptJudge:
    def __init__(self):
        self.statistics = STATISTICS

    def get_label(self, key: str, count: int) -> str:
        thresholds = self.statistics[key]
        keywords = QUANTIFIERS
        idx = min(int(np.digitize(count, thresholds, right=True)), len(keywords) - 1)
        return keywords[idx]

    def parse_prompt(self, prompt_str: str) -> Dict[str, str]:
        result = {}
        for part in prompt_str.split(","):
            part = part.strip()
            words = part.split(" ")
            if len(words) < 2:
                continue
            label = words[0]
            rest = " ".join(words[1:])
            if "moves" in rest and "air" not in rest:
                result["M"] = label
            elif "big jump" in rest:
                result["J"] = label
            elif "small jump" in rest:
                result["j"] = label
            elif "air" in rest:
                result["A"] = label
            elif "coin" in rest:
                result["C"] = label
        return result

    def analyze_and_report(self, level_str: str, target_prompt: str) -> Tuple[str, Dict[str, bool]]:
        lines = [l for l in level_str.strip().split('\n') if l.strip()]
        flattened = "".join(lines)

        counts = {
            "M": flattened.count("M"),
            "J": flattened.count("J"),
            "j": flattened.count("j"),
            "A": flattened.count("A"),
            "C": flattened.count("C"),
        }

        actual = {k: self.get_label(k, v) for k, v in counts.items()}
        target = self.parse_prompt(target_prompt)

        report = []
        report.append(f'使用的PROMPT: "{target_prompt}"')
        report.append("")
        report.append("=" * 50)
        report.append("路徑 Token 統計")
        report.append("=" * 50)
        report.append(f"M (moves):       {counts['M']:3d} -> [{actual['M']}]")
        report.append(f"J (big jumps):   {counts['J']:3d} -> [{actual['J']}]")
        report.append(f"j (small jumps): {counts['j']:3d} -> [{actual['j']}]")
        report.append(f"A (air moves):   {counts['A']:3d} -> [{actual['A']}]")
        report.append(f"C (coins):       {counts['C']:3d} -> [{actual['C']}]")
        report.append("-" * 50)

        gen_prompt = (
            f"{actual['M']} moves, {actual['J']} big jumps, "
            f"{actual['j']} small jumps, {actual['A']} air moves, "
            f"{actual['C']} coins collected"
        )
        report.append(f'實際對應 Prompt: "{gen_prompt}"')
        report.append("=" * 50)
        report.append("")
        report.append("差異比對結果:")
        report.append("-" * 50)

        results_bool = {}
        for key in ["M", "J", "j", "A", "C"]:
            if key not in target:
                continue
            t_val = target[key]
            a_val = actual[key]
            if t_val == a_val:
                status = "[符合]"
                diff = ""
                results_bool[key] = True
            else:
                status = "[不符]"
                diff = f"  (預期: {t_val} | 實際: {a_val})"
                results_bool[key] = False
            report.append(f"{status} {key:<10}: {a_val:<10} {diff}")

        report.append("-" * 50)
        report.append("\n\n")
        return "\n".join(report), results_bool


# ==========================================
# 3. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=GENERATE_COUNT)
    args = parser.parse_args()
    count = args.count

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Loading MarioLM from {CHECKPOINT_DIR} ...")
    mario_lm = MarioLM(lm_path=CHECKPOINT_DIR, tokenizer_path=TOKENIZER_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()

    positions_2d_full = build_positions_2d(context_len=700, height=14)

    judge = PromptJudge()
    report_path = os.path.join(OUTPUT_DIR, "prompt_check_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== PATH EXPERT BATCH GENERATION REPORT ===\n\n")

    print(f"\nStarting batch generation for {count} iterations...\n")

    for i in range(count):
        print(f"--- Iteration {i + 1}/{count} ---")

        p_M = random.choice(QUANTIFIERS)
        p_J = random.choice(QUANTIFIERS)
        p_j = random.choice(QUANTIFIERS)
        p_A = random.choice(QUANTIFIERS)
        p_C = random.choice(QUANTIFIERS)

        current_prompt = (
            f"{p_M} moves, {p_J} big jumps, {p_j} small jumps, "
            f"{p_A} air moves, {p_C} coins collected"
        )
        print(f"Prompt: {current_prompt}")

        generated_level = mario_lm.sample(
            prompts=[current_prompt],
            num_steps=1400,
            temperature=1.0,
            use_tqdm=True,
            level_height=14,
            positions_2d_NonExpand=positions_2d_full,
        )

        txt_path = os.path.join(OUTPUT_DIR, f"generated_level_{i}.txt")
        img_path = os.path.join(OUTPUT_DIR, f"generated_level_{i}.png")

        level_str = '\n'.join(generated_level.level)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(level_str)
            f.write(f'\n\n使用的PROMPT: "{current_prompt}"')

        if generated_level.img is not None:
            try:
                generated_level.img.save(img_path)
                print(f"Saved: {img_path}")
            except Exception as e:
                print(f"PNG save failed: {e}")

        print(f"Saved: {txt_path}")

        report_text, results_bool = judge.analyze_and_report(level_str, current_prompt)

        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"--- [Run #{i + 1}] ---\n")
            f.write(report_text)

        for key in results_bool:
            GLOBAL_STATS[key]["total"] += 1
            if results_bool[key]:
                GLOBAL_STATS[key]["correct"] += 1

        print("Analysis completed.\n")

    print("=" * 40)
    print("       FINAL ACCURACY STATISTICS       ")
    print("=" * 40)

    summary_lines = ["\n=== FINAL STATISTICS SUMMARY ===\n"]
    for key in ["M", "J", "j", "A", "C"]:
        stats = GLOBAL_STATS[key]
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        line = f"{key:<10}: {correct}/{total} correct ({acc:.1f}%)"
        print(line)
        summary_lines.append(line)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
