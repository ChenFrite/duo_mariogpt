import torch
import os

from mario_gpt_structure import MarioLM
from mario_gpt_structure.sampler import load_path_grid
from mario_gpt_structure.pipeline_pathexpert import PromptJudge

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 全域設定區
# ==========================================
GENERATE_COUNT = 100

TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
# 用 train.py 重新訓練出的 structure expert
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "Structure-GPT2-v2", "iteration_10000")
PATH_FILE      = os.path.join(_THIS_DIR, "test_path.txt")
OUTPUT_DIR     = os.path.join(_THIS_DIR, "same_path_test")


# ==========================================
# 2. 主程式
#    與 pipeline_pathexpert.py 相同機制：
#    - prompt 由 Prompter(sample_prompt=True) 產生（模板/字詞與訓練一致）
#    - 路徑 token 強制植入、輸出寬度 = 路徑檔寬度（sampler 內處理）
#    - PromptJudge 以 25 欄視窗、Prompter 同一套計數/門檻判定
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Loading MarioLM from {CHECKPOINT_DIR} ...")
    mario_lm = MarioLM(lm_path=CHECKPOINT_DIR, tokenizer_path=TOKENIZER_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()

    path_grid = load_path_grid(PATH_FILE, height=14)

    judge = PromptJudge(mario_lm.prompter)
    report_path = os.path.join(OUTPUT_DIR, "prompt_check_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== SAME-PATH STRUCTURE GENERATION REPORT ===\n\n")

    GLOBAL_STATS = {k: {"correct": 0, "total": 0, "window_frac": 0.0} for k in PromptJudge.FEATURES}

    print(f"\nStarting batch generation for {GENERATE_COUNT} iterations...\n")

    for i in range(GENERATE_COUNT):
        print(f"--- Iteration {i + 1}/{GENERATE_COUNT} ---")

        current_prompt, _, prompt_dict, _ = mario_lm.prompter(sample_prompt=True)
        target = {k: v.split()[0] for k, v in prompt_dict.items()}
        print(f"Prompt: {current_prompt}")

        generated_level = mario_lm.sample(
            path_grid=path_grid,
            prompts=[current_prompt],
            temperature=1.0,
            use_tqdm=False,
            level_height=14,
        )

        # --- 純結構圖 ---
        struct_txt = os.path.join(OUTPUT_DIR, f"structure_{i}.txt")
        struct_img = os.path.join(OUTPUT_DIR, f"structure_{i}.png")

        level_str = '\n'.join(generated_level.level)
        with open(struct_txt, "w", encoding="utf-8") as f:
            f.write(level_str)
            f.write(f'\n\n使用的PROMPT: "{current_prompt}"')
        print(f"Saved: {struct_txt}")

        if generated_level.img is not None:
            try:
                generated_level.img.save(struct_img)
                print(f"Saved: {struct_img}")
            except Exception as e:
                print(f"PNG save failed: {e}")

        report_text, match_frac = judge.analyze_and_report(generated_level.level, current_prompt, target)
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"--- [Run #{i + 1}] ---\n")
            f.write(report_text)

        for key, frac in match_frac.items():
            GLOBAL_STATS[key]["total"] += 1
            GLOBAL_STATS[key]["window_frac"] += frac
            if frac >= 0.5:
                GLOBAL_STATS[key]["correct"] += 1

        print("Analysis completed.\n")

    print("=" * 40)
    print("       FINAL ACCURACY STATISTICS       ")
    print("=" * 40)

    summary_lines = ["\n=== FINAL STATISTICS SUMMARY ===\n"]
    for key in PromptJudge.FEATURES:
        stats = GLOBAL_STATS[key]
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        win = (stats["window_frac"] / total * 100) if total > 0 else 0.0
        line = f"{key.capitalize():<10}: {correct}/{total} correct ({acc:.1f}%)  | 平均視窗符合率 {win:.1f}%"
        print(line)
        summary_lines.append(line)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
