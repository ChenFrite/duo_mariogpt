"""
批量掃描 multi_test 的 path，找「搜索價值明顯且生成不崩」的甜蜜點。

對每個 generated_level_i.txt：
  1. 統計 path 的跳躍密度（M/J/j/A 比例）
  2. 跑貪婪 vs 搜索，記錄 reward 和 gain
  3. 標記生成是否崩潰（reward 過低 / adherence 過差）

輸出：按「搜索 gain」排序的表，找到 gain 明顯的 path。

用法：放在 mario_gpt_structure/ 目錄，指向 multi_test 資料夾。
"""

from __future__ import annotations

import os, sys

import glob
import statistics as st

import torch

from mario_gpt_structure.lm import MarioLM
from mario_gpt_structure.sampler import GPTSampler, PATCH_WIDTH, FUTURE_WIDTH, load_path_grid
from mario_gpt_structure.sbs_propose import SBSMixin
from mario_gpt_structure.level_gen_env import LevelGenEnv, LevelState


class SBSSampler(SBSMixin, GPTSampler):
    pass


PATH_TOKEN_SET = set('MJjACLKWUkwRr')


def path_stats(path_grid):
    """統計 path 的跳躍密度。"""
    flat = "".join("".join(row) for row in path_grid)
    counts = {t: flat.count(t) for t in 'MJjA'}
    total_path = sum(counts.values())
    jump_ratio = (counts['J'] + counts['j']) / max(1, total_path)  # 跳躍佔比
    return counts, jump_ratio


# ---- 搜索 / 貪婪（同 compare_search_greedy）----
def rollout_greedy(env, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = env.propose_actions(s)
        s = env.step(s, cands[0])
    return env.terminal_reward(s)


def search_lookahead(env, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = env.propose_actions(s)
        best_c, best_r = None, -1e9
        for c in cands:
            s_next = env.step(s, c)
            rd = (env.terminal_reward(s_next) if env.is_terminal(s_next)
                  else rollout_greedy(env, s_next))
            if rd["reward"] > best_r:
                best_r, best_c = rd["reward"], c
        s = env.step(s, best_c)
    return env.terminal_reward(s)


def main():
    H = 14
    seg_cols = 2
    K = 4
    target_cols = int(os.environ.get("TARGET_COLS", "16"))
    N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
    MULTI_DIR = os.environ.get("MULTI_DIR", "multi_test")

    CKPT = os.environ.get("MARIO_CKPT", "mario_gpt_structure/Structure-GPT2-v2/iteration_10000")
    TOK = os.environ.get("MARIO_TOK", "shyamsn97/Mario-GPT2-700-context-length")
    print(f"Loading model {CKPT} ...")
    mario_lm = MarioLM(lm_path=CKPT, tokenizer_path=TOK)
    mario_lm.eval()
    sampler = SBSSampler(mario_lm, temperature=2.0, top_k=16)

    target_prompt = "some blocks, low elevation"
    with torch.no_grad():
        hidden = mario_lm.prompter.output_hidden(target_prompt)
    ehs = hidden.view(1, 1, -1).to(sampler.device)

    files = sorted(glob.glob(os.path.join(MULTI_DIR, "generated_level_*.txt")),
                   key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f))) or 0))

    results = []
    print(f"\nScanning {len(files)} paths, target_cols={target_cols}, "
          f"{N_TRIALS} trials each ...\n")

    for f in files:
        try:
            path_grid = load_path_grid(f, H)
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        W = len(path_grid[0])
        if W < PATCH_WIDTH + FUTURE_WIDTH:
            print(f"  skip {os.path.basename(f)}: width {W} too small")
            continue

        counts, jump_ratio = path_stats(path_grid)

        def make_env():
            return LevelGenEnv(
                sampler=sampler, path_grid=path_grid, target_prompt=target_prompt,
                encoder_hidden_states=ehs, seg_cols=seg_cols,
                target_cols=target_cols, H=H, K=K, use_mock_astar=True,
            )

        g_rews, s_rews = [], []
        for _ in range(N_TRIALS):
            g_rews.append(rollout_greedy(make_env(), make_env().reset())["reward"])
            s_rews.append(search_lookahead(make_env(), make_env().reset())["reward"])

        g_mean, s_mean = st.mean(g_rews), st.mean(s_rews)
        gain = s_mean - g_mean
        results.append({
            "file": os.path.basename(f),
            "jump_ratio": jump_ratio,
            "M": counts['M'], "J": counts['J'], "j": counts['j'], "A": counts['A'],
            "greedy": g_mean, "search": s_mean, "gain": gain,
        })
        print(f"  {os.path.basename(f):28s} jump={jump_ratio:.2f} "
              f"greedy={g_mean:.3f} search={s_mean:.3f} gain={gain:+.3f}")

    # ---- 匯總：按 gain 排序 ----
    print("\n=== Sorted by search gain (descending) ===")
    print(f"{'file':28s} {'jump':>5s} {'greedy':>7s} {'search':>7s} {'gain':>7s}")
    for r in sorted(results, key=lambda x: -x["gain"]):
        print(f"{r['file']:28s} {r['jump_ratio']:5.2f} "
              f"{r['greedy']:7.3f} {r['search']:7.3f} {r['gain']:+7.3f}")

    # ---- 找甜蜜點 ----
    print("\n=== Analysis ===")
    high_gain = [r for r in results if r["gain"] > 0.1]
    not_crashed = [r for r in results if r["greedy"] > 1.0]
    sweet = [r for r in results if r["gain"] > 0.1 and r["greedy"] > 1.0]

    print(f"Paths with clear search gain (>0.1): {len(high_gain)}")
    print(f"Paths not crashed (greedy>1.0): {len(not_crashed)}")
    print(f"甜蜜點（gain>0.1 且 greedy>1.0，難但不崩）: {len(sweet)}")
    if sweet:
        print("  → 這些 path 適合當搜索價值的實驗場景：")
        for r in sorted(sweet, key=lambda x: -x["gain"]):
            print(f"     {r['file']} (jump={r['jump_ratio']:.2f}, gain={r['gain']:+.3f})")
    else:
        print("  → 沒找到甜蜜點。可能全部饱和（干净）或全部崩溃（OOD）。")
        print("    建議：調 target_cols 增加難度，或手動構造中等 path。")


if __name__ == "__main__":
    main()

