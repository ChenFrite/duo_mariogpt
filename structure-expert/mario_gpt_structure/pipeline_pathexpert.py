#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import sys
import re
from typing import List, Tuple, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from mario_gpt_structure import MarioLM
from mario_gpt_structure.dataset import PATCH_WIDTH

# ==========================================
# 設定區
# ==========================================
TOKENIZER_PATH  = "shyamsn97/Mario-GPT2-700-context-length"
# 用 train.py 重新訓練出的 structure expert（舊的 Mario-GPT2-700-context-length_29 是 path-only 模型）
CHECKPOINT_DIR = os.path.join(_THIS_DIR, "Structure-GPT2-v2", "iteration_10000")

PATH_EXPERT_DIR = os.path.join(_THIS_DIR, "..", "..", "path", "mario_gpt_path", "multi_test")
OUTPUT_DIR      = os.path.join(_THIS_DIR, "pipeline_out")



# ==========================================
# PromptJudge
# ==========================================
class PromptJudge:
    """
    用與訓練時「完全相同」的 Prompter 計數/門檻來判定，且在相同尺度上判定：
    訓練時每個 prompt 描述的是一個 PATCH_WIDTH(25) 欄的 patch，
    所以這裡把生成關卡切成不重疊的 25 欄視窗，逐窗算 label，再和目標比對。
    某 feature 判為「符合」= 至少一半的視窗 label 與目標相同。
    """
    FEATURES = ["pipe", "enemy", "block", "rect", "coin", "elevation"]

    def __init__(self, prompter, window: int = PATCH_WIDTH):
        self.prompter = prompter
        self.window = window

    def window_labels(self, rows: List[str]) -> Dict[str, str]:
        flat = "".join(rows)
        p = self.prompter
        return {
            "pipe":      p.pipe_prompt(flat, rows)[1],
            "enemy":     p.enemy_prompt(flat, rows)[1],
            "block":     p.block_prompt(flat, rows)[1],
            "rect":      p.rect_prompt(flat, rows)[1],
            "coin":      p.coin_prompt(flat, rows)[1],
            "elevation": p.elevation_prompt(flat, rows)[1],
        }

    def analyze_and_report(self, level_rows: List[str], target_prompt: str,
                           target: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
        W = len(level_rows[0])
        starts = list(range(0, W - self.window + 1, self.window))
        windows = [self.window_labels([r[s:s + self.window] for r in level_rows]) for s in starts]

        report = [f'使用的PROMPT: "{target_prompt}"', "",
                  "=" * 60, f"逐窗判定（{len(windows)} 個 {self.window} 欄視窗，與訓練 patch 同尺度）", "=" * 60]
        for i, (s, lab) in enumerate(zip(starts, windows)):
            report.append(f"窗{i} col {s:3d}-{s + self.window - 1:3d}: " +
                          ", ".join(f"{k}={lab[k]}" for k in self.FEATURES))
        report += ["-" * 60, "差異比對結果（符合視窗比例）:", "-" * 60]

        match_frac: Dict[str, float] = {}
        for key in self.FEATURES:
            if key not in target or not windows:
                continue
            frac = sum(w[key] == target[key] for w in windows) / len(windows)
            match_frac[key] = frac
            status = "[符合]" if frac >= 0.5 else "[不符]"
            actual = ",".join(w[key] for w in windows)
            report.append(f"{status} {key.capitalize():<10}: 目標={target[key]:<7} 符合 {frac:4.0%}  實際=[{actual}]")
        report += ["-" * 60, "\n\n"]
        return "\n".join(report), match_frac


# ==========================================
# Path grid 解析
# ==========================================
def parse_pathexpert_file(filepath: str, height: int = 14) -> List[List[str]]:
    """
    讀取 path-expert 的 generated_level_N.txt。
    取前 14 行非空、非 PROMPT 行作為 path_grid。
    '使用的PROMPT:' 行直接捨棄。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    level_rows = []
    for ln in lines:
        if ln.startswith('使用的PROMPT:'):
            continue
        if ln.strip():
            level_rows.append(ln)
        if len(level_rows) == height:
            break

    assert len(level_rows) == height, \
        f"{filepath}: 期望 {height} 行，實際得到 {len(level_rows)} 行"

    W = len(level_rows[0])
    path_grid = [
        [level_rows[r][c] if c < len(level_rows[r]) else '-' for c in range(W)]
        for r in range(height)
    ]
    return path_grid


# ==========================================
# 主程式
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # 掃描所有 generated_level_N.txt，按 N 排序
    pattern = re.compile(r'^generated_level_(\d+)\.txt$')
    entries = []
    for fname in os.listdir(PATH_EXPERT_DIR):
        m = pattern.match(fname)
        if m:
            entries.append((int(m.group(1)), fname))
    entries.sort()

    if not entries:
        print(f"找不到任何 generated_level_N.txt：{PATH_EXPERT_DIR}")
        return
    print(f"找到 {len(entries)} 個路徑檔案。")

    print(f"Loading MarioLM from {CHECKPOINT_DIR} ...")
    mario_lm = MarioLM(lm_path=CHECKPOINT_DIR, tokenizer_path=TOKENIZER_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()

    judge = PromptJudge(mario_lm.prompter)
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== PIPELINE PATH-EXPERT -> STRUCTURE REPORT ===\n\n")

    GLOBAL_STATS = {k: {"correct": 0, "total": 0, "window_frac": 0.0} for k in PromptJudge.FEATURES}

    for idx, fname in entries:
        print(f"\n--- 處理 {fname} ---")

        filepath = os.path.join(PATH_EXPERT_DIR, fname)
        path_grid = parse_pathexpert_file(filepath, height=14)
        print(f"  path_grid: {len(path_grid)} 行 x {len(path_grid[0])} 欄")

        # 結構 prompt：用 Prompter 隨機抽，保證字詞/順序/模板與訓練時完全一致
        # （例如 blocks 只有 little/some/many、用 "rects"），避免 BART embedding 落在訓練分佈外
        current_prompt, _, prompt_dict, _ = mario_lm.prompter(sample_prompt=True)
        target = {k: v.split()[0] for k, v in prompt_dict.items()}
        print(f"  Prompt: {current_prompt}")

        generated_level = mario_lm.sample(
            path_grid=path_grid,
            prompts=[current_prompt],
            temperature=1.0,
            use_tqdm=False,
            level_height=14,
        )

        level_str = '\n'.join(generated_level.level)

        out_txt = os.path.join(OUTPUT_DIR, f"result_{idx}.txt")
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(level_str)
            f.write(f'\n\n使用的PROMPT: "{current_prompt}"')
        print(f"  Saved: {out_txt}")

        if generated_level.img is not None:
            try:
                out_img = os.path.join(OUTPUT_DIR, f"result_{idx}.png")
                generated_level.img.save(out_img)
                print(f"  Saved: {out_img}")
            except Exception as e:
                print(f"  PNG save failed: {e}")

        report_text, match_frac = judge.analyze_and_report(generated_level.level, current_prompt, target)
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"--- [{fname}] ---\n")
            f.write(report_text)

        for key, frac in match_frac.items():
            GLOBAL_STATS[key]["total"] += 1
            GLOBAL_STATS[key]["window_frac"] += frac
            if frac >= 0.5:
                GLOBAL_STATS[key]["correct"] += 1

    print("\n" + "=" * 40)
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

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\nFull report saved to {report_path}")


if __name__ == '__main__':
    main()
