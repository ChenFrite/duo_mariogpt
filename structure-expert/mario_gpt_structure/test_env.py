"""
跑通 LevelGenEnv 的最小測試：不用完整 Gumbel 搜索，先用「貪婪選 logprob 最高」
把一條 level 生成到終端，算 reward。驗證 SBS → env → reward 整條鏈路通。

阶段一：use_mock_astar=True（跳過 5 秒 A*），target_cols 設短。
驗證數值時：改 use_mock_astar=False。
"""

from __future__ import annotations

import os, sys

import torch

from mario_gpt_structure.lm import MarioLM       # 依實際路徑調整
from mario_gpt_structure.sampler import GPTSampler, PATCH_WIDTH, FUTURE_WIDTH, load_path_grid
from mario_gpt_structure.sbs_propose import SBSMixin
from mario_gpt_structure.level_gen_env import LevelGenEnv


class SBSSampler(SBSMixin, GPTSampler):
    pass


def main():
    H = 14
    seg_cols = 2
    K = 4
    target_cols = 8          # 阶段一：短關卡（8 欄）先跑通
    temperature = 2.0

    CKPT = os.environ.get("MARIO_CKPT", "Mario-GPT2-700-context-length")
    print(f"Loading {CKPT} ...")
    mario_lm = MarioLM(lm_path=CKPT, tokenizer_path=CKPT)
    mario_lm.eval()
    sampler = SBSSampler(mario_lm, temperature=temperature, top_k=16)

    # path_grid：要夠寬（至少 PATCH_WIDTH+FUTURE_WIDTH+target_cols）
    PATH_FILE = os.environ.get("PATH_FILE", "")
    if PATH_FILE and os.path.exists(PATH_FILE):
        path_grid = load_path_grid(PATH_FILE, H)
    else:
        W = PATCH_WIDTH + FUTURE_WIDTH + target_cols + 2
        path_grid = [['-'] * W for _ in range(H)]
    print(f"path_grid width = {len(path_grid[0])}")

    # Use the target prompt itself to condition generation.
    target_prompt = "some blocks, low elevation"
    with torch.no_grad():
        hidden = mario_lm.prompter.output_hidden(target_prompt)
        encoder_hidden_states = hidden.view(1, 1, -1).to(sampler.device)

    env = LevelGenEnv(
        sampler=sampler,
        path_grid=path_grid,
        target_prompt=target_prompt,
        encoder_hidden_states=encoder_hidden_states,
        seg_cols=seg_cols,
        target_cols=target_cols,
        H=H, K=K,
        use_mock_astar=True,      # 阶段一
    )

    # ---- 貪婪 rollout：每步 propose K 個，選 logprob 最高的 ----
    state = env.reset()
    step_i = 0
    while not env.is_terminal(state):
        candidates = env.propose_actions(state)
        best = candidates[0]   # propose_segment 已按 logprob 降序
        print(f"[step {step_i}] num_cols={state.num_cols} → "
              f"chose cand logprob={best.logprob:.3f}, {len(candidates)} candidates")
        state = env.step(state, best)
        step_i += 1

    print(f"\nReached terminal at {state.num_cols} cols.")

    # ---- 終端 reward ----
    rew = env.terminal_reward(state)
    print("\n=== Terminal Reward ===")
    for k, v in rew.items():
        print(f"  {k:12s} = {v:.4f}")

    # ---- 印出生成的 level ----
    print("\n=== Generated Level ===")
    rows = env._to_level_rows(state)
    for r in rows:
        print(r)

    # ---- 可選：用真 A* 對照 mock ----
    if os.environ.get("REAL_ASTAR", "0") == "1":
        print("\n=== Real A* check ===")
        from mario_gpt_structure.astar_simulator import run_astar_solvable
        real = run_astar_solvable(rows)
        print(f"  real A* solvable = {real}  (mock said {rew['solvable']==1.0})")


if __name__ == "__main__":
    main()
