"""
批次：search vs greedy，跑成分布而非單點。
對齊重整後 mario_gpt_structure + Structure-GPT2-v2 + sbs_propose。

對「多條 path × 多個隨機 prompt」各跑 greedy 與 1-step-lookahead search，輸出：
  - 聚合：mean greedy / mean search / mean gain / win-rate（search ≥ greedy 的比例）
  - per-feature 分解：greedy vs search 各屬性「命中目標 label」的比例
    （看搜索到底把哪些屬性的可控性補回來——預期 enemy/rect/coin）
  - 每個 (K, path, prompt) 一列的 CSV
  - K-sweep：K_LIST="2,4,8" 一次掃多個預算，畫 gain-vs-cost

批次用 mock A*（快、可大量跑）；solvable 在 mock 下近乎恆 1，不是區分訊號，
量化結論看 adh_struct 的 gain 與 win-rate。真 A* 留給最後一輪小樣本單獨驗證 solvable。

放在 structure-expert/mario_gpt_structure/ 下：
  export MARIO_CKPT=Structure-GPT2-v2/iteration_10000
  export PATH_DIR=../../path/mario_gpt_path/multi_test
  export N_PROMPTS=4 TARGET_COLS=16 SEG_COLS=2 K_LIST=4 N_TRIALS=1
  export LAMBDA_S=1.0 LAMBDA_P=0.25 SEED=0
  python batch_search_vs_greedy.py
"""
from __future__ import annotations

import csv
import glob
import os
import random
import statistics as st
from typing import Dict, List

import torch

from mario_gpt_structure import MarioLM
from mario_gpt_structure.sampler import GPTSampler, load_path_grid
from mario_gpt_structure.sbs_propose import SBSMixin
from mario_gpt_structure.level_gen_env import LevelGenEnv

FEATURES = ["pipe", "enemy", "block", "rect", "coin", "elevation"]
_FEAT_NOUN = [("pipe", "pipe"), ("enem", "enemy"), ("block", "block"),
              ("rect", "rect"), ("coin", "coin"), ("elevation", "elevation")]


def target_labels(prompt: str) -> Dict[str, str]:
    """把 "many pipes, no enemies, ..., high elevation" 解析成 {feature: 量化詞}（同 prompter 回傳的 label）。"""
    out = {}
    for item in prompt.split(","):
        words = item.strip().lower().split()
        if len(words) < 2:
            continue
        for noun, feat in _FEAT_NOUN:
            if words[1].startswith(noun):
                out[feat] = words[0]
                break
    return out


class SBSSampler(SBSMixin, GPTSampler):
    pass


class ProposeCounter:
    def __init__(self, env):
        self.env = env
        self.calls = 0
        self.candidates = 0

    def propose(self, state):
        self.calls += 1
        cands = self.env.propose_actions(state)
        self.candidates += len(cands)
        return cands


def rollout_greedy(env, counter, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = counter.propose(s)
        s = env.step(s, cands[0])
    return env.terminal_reward(s), s


def search_lookahead(env, counter, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = counter.propose(s)
        best_c, best_r = None, -1e9
        for c in cands:
            s_next = env.step(s, c)
            if env.is_terminal(s_next):
                rd = env.terminal_reward(s_next)
            else:
                rd, _ = rollout_greedy(env, counter, s_next)
            if rd["reward"] > best_r:
                best_r, best_c = rd["reward"], c
        s = env.step(s, best_c)
    return env.terminal_reward(s), s


def actual_labels(env, rows) -> Dict[str, str]:
    flat = "".join(rows)
    p = env.sampler.mario_lm.prompter
    return {
        "pipe": p.pipe_prompt(flat, rows)[1],
        "enemy": p.enemy_prompt(flat, rows)[1],
        "block": p.block_prompt(flat, rows)[1],
        "rect": p.rect_prompt(flat, rows)[1],
        "coin": p.coin_prompt(flat, rows)[1],
        "elevation": p.elevation_prompt(flat, rows)[1],
    }


def feature_match(env, state) -> Dict[str, int]:
    """每個目標屬性：actual label 是否 == target label（1/0）。"""
    rows = env._to_level_rows(state)
    act = actual_labels(env, rows)
    out = {}
    for feat, tgt in target_labels(env.target_prompt).items():
        out[feat] = int(act.get(feat) == tgt)
    return out


def main():
    H = 14
    CKPT = os.environ.get("MARIO_CKPT", "Structure-GPT2-v2/iteration_10000")
    TOK = os.environ.get("MARIO_TOK", "shyamsn97/Mario-GPT2-700-context-length")
    PATH_DIR = os.environ.get("PATH_DIR", "../../path/mario_gpt_path/multi_test")
    PATH_FILE = os.environ.get("PATH_FILE", "test_path.txt")
    N_PATHS = int(os.environ.get("N_PATHS", "10"))
    N_PROMPTS = int(os.environ.get("N_PROMPTS", "4"))
    N_TRIALS = int(os.environ.get("N_TRIALS", "1"))
    TARGET_COLS = int(os.environ.get("TARGET_COLS", "16"))
    SEG_COLS = int(os.environ.get("SEG_COLS", "2"))
    K_LIST = [int(x) for x in os.environ.get("K_LIST", "4").split(",") if x.strip()]
    TEMP = float(os.environ.get("TEMP", "1.0"))
    TOPK = int(os.environ.get("TOPK", "16"))
    LAMBDA_S = float(os.environ.get("LAMBDA_S", "1.0"))
    LAMBDA_P = float(os.environ.get("LAMBDA_P", "0.25"))
    SEED = int(os.environ.get("SEED", "0"))
    OUT_CSV = os.environ.get("OUT_CSV", "batch_results.csv")

    _here = os.path.dirname(os.path.abspath(__file__))
    ckpt = CKPT if os.path.isabs(CKPT) else os.path.join(_here, CKPT)

    # 收集 path 檔
    path_dir = PATH_DIR if os.path.isabs(PATH_DIR) else os.path.join(_here, PATH_DIR)
    path_files = sorted(
        glob.glob(os.path.join(path_dir, "generated_level_*.txt")),
        key=lambda f: int("".join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0),
    )[:N_PATHS]
    if not path_files:
        pf = PATH_FILE if os.path.isabs(PATH_FILE) else os.path.join(_here, PATH_FILE)
        path_files = [pf]
    print(f"paths: {len(path_files)} 條")

    print(f"Loading MarioLM: {ckpt}")
    mario_lm = MarioLM(lm_path=ckpt, tokenizer_path=TOK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()
    sampler = SBSSampler(mario_lm, temperature=TEMP, top_k=TOPK)

    # 預抽 N_PROMPTS 個 prompt（seeded，跨所有 path 共用同一組，便於比較）
    random.seed(SEED)
    prompts = [mario_lm.prompter(sample_prompt=True)[0] for _ in range(N_PROMPTS)]
    print(f"prompts: {N_PROMPTS} 個（seed={SEED}）")
    for p in prompts:
        print(f"   - {p.strip()}")

    # 預載 path_grid
    grids = {f: load_path_grid(f, H) for f in path_files}

    rows_csv: List[dict] = []
    print(f"\nK_LIST={K_LIST}  TARGET_COLS={TARGET_COLS}  SEG_COLS={SEG_COLS}  "
          f"N_TRIALS={N_TRIALS}  λs={LAMBDA_S} λp={LAMBDA_P}\n")

    for K in K_LIST:
        g_rew, s_rew, gains, wins, g_cost, s_cost = [], [], [], [], [], []
        g_struct, s_struct = [], []
        feat_g = {f: [] for f in FEATURES}
        feat_s = {f: [] for f in FEATURES}

        for pf in path_files:
            grid = grids[pf]
            for prompt in prompts:
                with torch.no_grad():
                    ehs = mario_lm.prompter.output_hidden(prompt).view(1, 1, -1).to(device)

                def make_env():
                    return LevelGenEnv(
                        sampler=sampler, path_grid=grid, target_prompt=prompt,
                        encoder_hidden_states=ehs, seg_cols=SEG_COLS,
                        target_cols=TARGET_COLS, H=H, K=K,
                        lambda_s=LAMBDA_S, lambda_p=LAMBDA_P, use_mock_astar=True,
                    )  # temperature 由 sampler（SBSSampler(temperature=TEMP)）決定

                for _ in range(N_TRIALS):
                    env = make_env(); ctr = ProposeCounter(env)
                    gr, gs = rollout_greedy(env, ctr, env.reset())
                    gcost = ctr.candidates
                    env = make_env(); ctr = ProposeCounter(env)
                    sr, ss = search_lookahead(env, ctr, env.reset())
                    scost = ctr.candidates

                    g_rew.append(gr["reward"]); s_rew.append(sr["reward"])
                    g_struct.append(gr["adh_struct"]); s_struct.append(sr["adh_struct"])
                    gains.append(sr["reward"] - gr["reward"])
                    wins.append(1 if sr["reward"] >= gr["reward"] + 1e-9 else 0)
                    g_cost.append(gcost); s_cost.append(scost)

                    gm = feature_match(env, gs)
                    sm = feature_match(env, ss)
                    for f in target_labels(prompt):
                        feat_g[f].append(gm[f]); feat_s[f].append(sm[f])

                    rows_csv.append({
                        "K": K, "path": os.path.basename(pf), "prompt": prompt.strip(),
                        "greedy_reward": round(gr["reward"], 4),
                        "search_reward": round(sr["reward"], 4),
                        "gain": round(sr["reward"] - gr["reward"], 4),
                        "win": wins[-1],
                        "greedy_adh_struct": round(gr["adh_struct"], 4),
                        "search_adh_struct": round(sr["adh_struct"], 4),
                        "greedy_cost": gcost, "search_cost": scost,
                        **{f"g_{f}": gm.get(f, "") for f in FEATURES},
                        **{f"s_{f}": sm.get(f, "") for f in FEATURES},
                    })

        n = len(g_rew)
        print("=" * 64)
        print(f"K={K}   (n = {n} 組 path×prompt×trial)")
        print("-" * 64)
        print(f"  greedy reward : {st.mean(g_rew):.3f} ± {st.pstdev(g_rew):.3f}")
        print(f"  search reward : {st.mean(s_rew):.3f} ± {st.pstdev(s_rew):.3f}")
        print(f"  mean gain     : {st.mean(gains):+.3f}")
        print(f"  win-rate      : {100*st.mean(wins):.1f}%  (search ≥ greedy)")
        print(f"  adh_struct    : greedy {st.mean(g_struct):.3f} → search {st.mean(s_struct):.3f}")
        print(f"  cost (cand)   : greedy {st.mean(g_cost):.0f}  search {st.mean(s_cost):.0f}  "
              f"({st.mean(s_cost)/max(1,st.mean(g_cost)):.1f}x)")
        print("  per-feature 命中目標 label 比例 (greedy → search):")
        for f in FEATURES:
            if feat_g[f]:
                print(f"     {f:10s}: {100*st.mean(feat_g[f]):5.1f}% → {100*st.mean(feat_s[f]):5.1f}%")
        print("=" * 64 + "\n")

    # 寫 CSV
    out_csv = OUT_CSV if os.path.isabs(OUT_CSV) else os.path.join(_here, OUT_CSV)
    if rows_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"CSV 寫出：{out_csv}  ({len(rows_csv)} 列)")


if __name__ == "__main__":
    main()
