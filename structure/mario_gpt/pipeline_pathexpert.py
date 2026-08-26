#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import sys
import re
import random
import numpy as np
from typing import List, Tuple, Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from mario_gpt import MarioLM

# ==========================================
# 設定區
# ==========================================
TOKENIZER_PATH  = "shyamsn97/Mario-GPT2-700-context-length"
CHECKPOINT_DIR  = "./M_multiround_round10/iteration_10000"

PATH_EXPERT_DIR = os.path.join(_THIS_DIR, "..", "..", "path", "mario_gpt", "multi_test")
OUTPUT_DIR      = os.path.join(_THIS_DIR, "pipeline_out")

QUANTIFIERS = ["no", "little", "some", "many"]
ELEVATIONS  = ["low", "high"]


# ==========================================
# PromptJudge
# ==========================================
class PromptJudge:
    def __init__(self):
        self.statistics = {
            "pipe":  np.array([0.0, 1.0, 3.0]),
            "enemy": np.array([0.0, 1.0, 3.0]),
            "block": np.array([0.0, 7.0, 20.0]),
            "coin":  np.array([0.0, 3.0, 8.0]),
            "rect":  np.array([0.0, 1.0, 2.0]),
        }

    def get_keywords(self, key):
        return ["no", "little", "some", "many"]

    def count_pipes(self, flattened: str) -> int:
        return flattened.count("<>")

    def count_enemies(self, flattened: str) -> int:
        return flattened.count("E") + flattened.count("B")

    def count_blocks(self, flattened: str) -> int:
        return int(np.sum([flattened.count(c) for c in ["S", "?", "Q"]]))

    def count_coins(self, flattened: str) -> int:
        return flattened.count("o")

    def _find_rectangle_size(self, grid, row, col, H, W) -> Tuple[int, int]:
        width = 0
        while col + width < W and grid[row][col + width] == 'T':
            width += 1
        if width < 2:
            return 0, 0
        height = 0
        while row + height < H and grid[row + height][col] == 'T':
            height += 1
        if height < 2:
            return 0, 0
        if row + height - 1 >= H:
            return 0, 0
        for c in range(col, col + width):
            if grid[row + height - 1][c] != 'T':
                return 0, 0
        if col + width - 1 >= W:
            return 0, 0
        for r in range(row, row + height):
            if grid[r][col + width - 1] != 'T':
                return 0, 0
        return height, width

    def count_rects(self, flattened: str) -> int:
        H = 14
        if len(flattened) == 0 or len(flattened) % H != 0:
            return 0
        W = len(flattened) // H
        grid = [list(flattened[i * W:(i + 1) * W]) for i in range(H)]
        visited = set()
        count = 0
        for row in range(H):
            for col in range(W):
                if (row, col) not in visited and grid[row][col] == 'T':
                    h, w = self._find_rectangle_size(grid, row, col, H, W)
                    if h > 0 and w > 0:
                        count += 1
                        for r in range(row, row + h):
                            for c in range(col, col + w):
                                visited.add((r, c))
        return count

    def get_label(self, category: str, count: int) -> str:
        thresholds = self.statistics[category]
        keywords = self.get_keywords(category)
        idx = min(int(np.digitize(count, thresholds, right=True)), len(keywords) - 1)
        return keywords[idx]

    def check_elevation(self, lines: List[str]) -> str:
        for t in lines[:6]:
            if "X" in t or "<" in t or ">" in t:
                return "high"
        return "low"

    def parse_prompt(self, prompt_str: str) -> Dict[str, str]:
        result = {}
        for part in prompt_str.split(","):
            words = part.strip().split(" ")
            if len(words) >= 2:
                label, key = words[0], words[1]
                if "pipe" in key:
                    result["pipe"] = label
                elif "block" in key:
                    result["block"] = label
                elif "coin" in key:
                    result["coin"] = label
                elif "enem" in key:
                    result["enemy"] = label
                elif "rect" in key:
                    result["rect"] = label
                elif "elev" in key:
                    result["elevation"] = label
        return result

    def analyze_and_report(self, level_str: str, target_prompt: str) -> Tuple[str, Dict[str, bool]]:
        lines = [l for l in level_str.strip().split('\n') if l.strip()]
        flattened = "".join(lines)

        n_pipes   = self.count_pipes(flattened)
        n_enemies = self.count_enemies(flattened)
        n_blocks  = self.count_blocks(flattened)
        n_coins   = self.count_coins(flattened)
        n_rects   = self.count_rects(flattened)

        actual = {
            "pipe":      self.get_label("pipe",  n_pipes),
            "enemy":     self.get_label("enemy", n_enemies),
            "block":     self.get_label("block", n_blocks),
            "coin":      self.get_label("coin",  n_coins),
            "rect":      self.get_label("rect",  n_rects),
            "elevation": self.check_elevation(lines),
        }

        target = self.parse_prompt(target_prompt)

        report = []
        report.append(f'使用的PROMPT: "{target_prompt}"')
        report.append("")
        report.append("=" * 50)
        report.append("關卡內容統計")
        report.append("=" * 50)
        report.append(f"水管 (Pipes):    {n_pipes:3d} -> [{actual['pipe']}]")
        report.append(f"敵人 (Enemies):  {n_enemies:3d} -> [{actual['enemy']}]")
        report.append(f"磚塊 (Blocks):   {n_blocks:3d} -> [{actual['block']}]")
        report.append(f"金幣 (Coins):    {n_coins:3d} -> [{actual['coin']}]")
        report.append(f"矩形 (Rects):    {n_rects:3d} -> [{actual['rect']}]")
        report.append(f"海拔 (Height):        -> [{actual['elevation']}]")
        report.append("-" * 50)

        gen_prompt = (
            f"{actual['rect']} rect, {actual['coin']} coins, "
            f"{actual['pipe']} pipes, {actual['enemy']} enemies, "
            f"{actual['block']} blocks, {actual['elevation']} elevation"
        )
        report.append(f'實際對應 Prompt: "{gen_prompt}"')
        report.append("=" * 50)
        report.append("")
        report.append("差異比對結果:")
        report.append("-" * 50)

        results_bool = {}
        for key in ["pipe", "enemy", "block", "coin", "rect", "elevation"]:
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
            report.append(f"{status} {key.capitalize():<10}: {a_val:<10} {diff}")

        report.append("-" * 50)
        report.append("\n\n")
        return "\n".join(report), results_bool


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

    judge = PromptJudge()
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== PIPELINE PATH-EXPERT -> STRUCTURE REPORT ===\n\n")

    GLOBAL_STATS = {
        "pipe":      {"correct": 0, "total": 0},
        "enemy":     {"correct": 0, "total": 0},
        "block":     {"correct": 0, "total": 0},
        "coin":      {"correct": 0, "total": 0},
        "rect":      {"correct": 0, "total": 0},
        "elevation": {"correct": 0, "total": 0},
    }

    for idx, fname in entries:
        print(f"\n--- 處理 {fname} ---")

        filepath = os.path.join(PATH_EXPERT_DIR, fname)
        path_grid = parse_pathexpert_file(filepath, height=14)
        print(f"  path_grid: {len(path_grid)} 行 x {len(path_grid[0])} 欄")

        # 結構生成 prompt 隨機產生
        p_rect  = random.choice(QUANTIFIERS)
        p_coin  = random.choice(QUANTIFIERS)
        p_pipe  = random.choice(QUANTIFIERS)
        p_enemy = random.choice(QUANTIFIERS)
        p_block = random.choice(QUANTIFIERS)
        p_elev  = random.choice(ELEVATIONS)
        current_prompt = (
            f"{p_pipe} pipes, {p_enemy} enemies, {p_block} blocks, {p_coin} coins, "
            f" {p_rect} rect, {p_elev} elevation"
        )
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

        report_text, results_bool = judge.analyze_and_report(level_str, current_prompt)
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"--- [{fname}] ---\n")
            f.write(report_text)

        for key in results_bool:
            GLOBAL_STATS[key]["total"] += 1
            if results_bool[key]:
                GLOBAL_STATS[key]["correct"] += 1

    print("\n" + "=" * 40)
    print("       FINAL ACCURACY STATISTICS       ")
    print("=" * 40)
    summary_lines = ["\n=== FINAL STATISTICS SUMMARY ===\n"]
    for key in ["pipe", "enemy", "block", "coin", "rect", "elevation"]:
        stats = GLOBAL_STATS[key]
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total > 0 else 0.0
        line = f"{key.capitalize():<10}: {correct}/{total} correct ({acc:.1f}%)"
        print(line)
        summary_lines.append(line)

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    print(f"\nFull report saved to {report_path}")


if __name__ == '__main__':
    main()
