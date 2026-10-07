"""
地基 smoke test：用純 sampler.sample() 生成一條 level，確認 checkpoint 可用、
生成有真實結構（非全 path token / 空白）、reward 分項合理。

放在 structure-expert/mario_gpt_structure/ 執行。
用正確的 checkpoint（自帶 tokenizer 的那個）：
  export MARIO_CKPT=Structure-GPT2-v2/iteration_10000
  export MARIO_TOK=shyamsn97/Mario-GPT2-700-context-length
  export PATH_FILE=test_path.txt
  python diagnose_sampler.py
"""

from __future__ import annotations

import os, sys

import torch
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

from mario_gpt_structure.lm import MarioLM
from mario_gpt_structure.sampler import GPTSampler, load_path_grid


def _mock_solvable(level_rows, H=14):
    """對齊 level_gen_env.mock_solvable：最底列斷崖 <=4 當可通關。"""
    ground = level_rows[-1]
    max_gap = cur = 0
    for ch in ground:
        if ch == '-':
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    return max_gap <= 4


def _count_structure(level_rows):
    """統計結構物件（非 path、非空白），確認模型真的生成了結構。"""
    flat = "".join(level_rows)
    struct_chars = {
        'X': 'ground/unbreakable', 'S': 'breakable', '?': 'question',
        'Q': 'question', 'o': 'coin', 'E': 'enemy', 'B': 'cannon',
        '<': 'pipe', '>': 'pipe', '[': 'pipe', ']': 'pipe', 'T': 'rect',
    }
    counts = {}
    for ch, name in struct_chars.items():
        c = flat.count(ch)
        if c > 0:
            counts[ch] = c
    total_struct = sum(counts.values())
    total_tiles = len(flat)
    return counts, total_struct, total_tiles


def main():
    H = 14
    CKPT = os.environ.get("MARIO_CKPT", "Structure-GPT2-v2/iteration_10000")
    TOK = os.environ.get("MARIO_TOK", "shyamsn97/Mario-GPT2-700-context-length")   # structure checkpoint 不含 tokenizer 檔
    PATH_FILE = os.environ.get("PATH_FILE", "test_path.txt")

    print(f"Loading model   : {CKPT}")
    print(f"Loading tokenizer: {TOK}")
    mario_lm = MarioLM(lm_path=CKPT, tokenizer_path=TOK)
    mario_lm.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mario_lm.eval()
    seed = int(os.environ.get("SEED", "0"))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Device: {mario_lm.device}; seed: {seed}")

    sampler = GPTSampler(mario_lm, temperature=2.0, top_k=16)

    print(f"\nGenerating with pure sampler.sample(), path={PATH_FILE} ...")
    target_prompt = os.environ.get("TARGET_PROMPT", "some blocks, low elevation")
    print(f"STRUCT prompt: {target_prompt}")
    raw_ids = []
    next_token = sampler._next_token
    def tracked_next_token(*args, **kwargs):
        token = next_token(*args, **kwargs)
        raw_ids.append(token)
        return token
    sampler._next_token = tracked_next_token
    out = sampler.sample(path_file=PATH_FILE, prompts=[target_prompt], use_tqdm=False)
    output_dir = Path(os.environ.get("OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    out.save(str(output_dir / "sampler_baseline.txt"))
    print("Saved sampler_baseline.txt")
    if out.img is not None:
        out.img.save(output_dir / "sampler_baseline.png")
    from collections import Counter
    raw_counts = dict(Counter(mario_lm.tokenizer.decode([i]) for i in raw_ids))
    print("Pre-forced model token counts:", raw_counts)
    print("Final tile counts:", dict(Counter("".join(out.level))))
    path_grid = load_path_grid(PATH_FILE, H)
    from mario_gpt_structure.sampler import PATH_TOKENS
    free_counts = Counter()
    forced_mismatches = 0
    for r, row in enumerate(out.level):
        for c, ch in enumerate(row):
            if path_grid[r][c] in PATH_TOKENS:
                forced_mismatches += ch != path_grid[r][c]
            else:
                free_counts[ch] += 1
    print("Free-position tile counts:", dict(free_counts))
    print("Forced-position mismatches:", forced_mismatches)

    print("\n=== Generated level ===")
    for row in out.level:
        print(row)

    # ---- 結構診斷：確認不是全 path token / 空白 ----
    counts, total_struct, total_tiles = _count_structure(out.level)
    print("\n=== Structure diagnosis ===")
    print(f"  structure tiles: {total_struct} / {total_tiles} "
          f"({100*total_struct/max(1,total_tiles):.1f}%)")
    print(f"  breakdown: {counts}")
    if total_struct == 0:
        print("  [!!] 結構全 0 —— 模型可能沒載入對、或 checkpoint 有問題")
    else:
        print("  [OK] 有真實結構生成")

    # ---- reward 分項 ----
    try:
        from mario_gpt_structure.adherence import terminal_reward
    except ImportError:
        print("\n(adherence.py 不在，略過 reward 評分)")
        return

    N = int(os.environ.get("SCORE_COLS", "16"))
    level_rows = [row[:N] for row in out.level]
    path_grid = load_path_grid(PATH_FILE, H)
    target_prompt = os.environ.get("TARGET_PROMPT", "some blocks, low elevation")

    use_real = os.environ.get("REAL_ASTAR", "0") == "1"
    if use_real:
        from mario_gpt_structure.astar_simulator import run_astar_solvable
        solvable = run_astar_solvable(level_rows)
    else:
        solvable = _mock_solvable(level_rows)

    rew = terminal_reward(
        level_rows=level_rows, target_prompt=target_prompt,
        path_grid=path_grid, prompter=mario_lm.prompter,
        astar_solvable=solvable, H=H,
    )
    print(f"\n=== Baseline reward (first {N} cols, "
          f"{'real' if use_real else 'mock'} A*) ===")
    for k, v in rew.items():
        print(f"  {k:12s} = {v:.4f}")

    summary = {
        "checkpoint": str(Path(CKPT).resolve()), "tokenizer": TOK,
        "path": str(Path(PATH_FILE).resolve()), "prompt": target_prompt, "seed": seed,
        "device": str(mario_lm.device), "structure_tiles": total_struct,
        "total_tiles": total_tiles, "structure_counts": counts,
        "pre_forced_token_counts": raw_counts, "free_token_counts": dict(free_counts),
        "forced_mismatches": forced_mismatches, "score_cols": N, "reward": rew,
        "astar_mode": "real" if use_real else "mock", "passed_nonzero_structure": total_struct > 0,
    }
    (output_dir / "diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n判斷：")
    print("  非全零結構只是必要條件，reward 接近 2.0 並非驗收要求")
    print("  結構全 0 → 生成／模型／資料問題待診斷，不執行難度掃描")
    if total_struct == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
