"""
搜索 vs 貪婪：在短關卡（預設 8 欄）驗證「search 贏 greedy」這個核心論點。
對齊重整後的 mario_gpt_structure 套件 + Structure-GPT2-v2 + sbs_propose。

兩種策略（每段都先用 SBS propose K 個候選）：
  greedy  : 每段取 candidates[0]（log π_θ 最高）直接 step。
  search  : 每段對 K 個候選各做一次 greedy rollout 到 terminal，
            取 terminal reward 最高的候選 step（1-step lookahead）。

兩者每一步看到的是「同一組 K 候選」，差別只在「選 logprob 最高」vs「選 rollout reward 最高」，
所以差異純粹來自搜索的選擇準則，apples-to-apples。

放在 structure-expert/mario_gpt_structure/ 下執行：
  export MARIO_CKPT=Structure-GPT2-v2/iteration_10000
  export PATH_FILE=test_path.txt
  export TARGET_COLS=8 SEG_COLS=2 K=4 N_TRIALS=5
  python compare_search_greedy.py
"""
from __future__ import annotations

import os
import statistics as st

import torch

from mario_gpt_structure import MarioLM
from mario_gpt_structure.sampler import GPTSampler, load_path_grid
from mario_gpt_structure.sbs_propose import SBSMixin
from mario_gpt_structure.level_gen_env import LevelGenEnv


class SBSSampler(SBSMixin, GPTSampler):
    """GPTSampler + SBS 候選提案。"""
    pass


# ---- propose 次數計數器（成本指標）----
class ProposeCounter:
    def __init__(self, env: LevelGenEnv):
        self.env = env
        self.calls = 0          # propose 呼叫次數
        self.candidates = 0     # 累計候選數（≈ LLM 展開量）
        self._orig = env.propose_actions

    def propose(self, state):
        self.calls += 1
        cands = self._orig(state)
        self.candidates += len(cands)
        return cands


def rollout_greedy(env: LevelGenEnv, counter: ProposeCounter, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = counter.propose(s)
        s = env.step(s, cands[0])      # logprob 最高
    return env.terminal_reward(s)


def search_lookahead(env: LevelGenEnv, counter: ProposeCounter, state):
    s = state.clone()
    while not env.is_terminal(s):
        cands = counter.propose(s)
        best_c, best_r = None, -1e9
        for c in cands:
            s_next = env.step(s, c)
            rd = (env.terminal_reward(s_next) if env.is_terminal(s_next)
                  else rollout_greedy(env, counter, s_next))
            if rd["reward"] > best_r:
                best_r, best_c = rd["reward"], c
        s = env.step(s, best_c)
    return env.terminal_reward(s)


def main():
    H = 14
    CKPT = os.environ.get("MARIO_CKPT", "Structure-GPT2-v2/iteration_10000")
    TOK = os.environ.get("MARIO_TOK", "shyamsn97/Mario-GPT2-700-context-length")
    PATH_FILE = os.environ.get("PATH_FILE", "test_path.txt")
    TARGET_COLS = int(os.environ.get("TARGET_COLS", "8"))
    SEG_COLS = int(os.environ.get("SEG_COLS", "2"))
    K = int(os.environ.get("K", "4"))
    N_TRIALS = int(os.environ.get("N_TRIALS", "5"))
    TEMP = float(os.environ.get("TEMP", "1.0"))
    TOPK = int(os.environ.get("TOPK", "16"))
    LAMBDA_S = float(os.environ.get("LAMBDA_S", "0.5"))
    LAMBDA_P = float(os.environ.get("LAMBDA_P", "0.5"))
    REWARD_FEATURES = [x for x in os.environ.get(
        "REWARD_FEATURES", "pipe,enemy,block,coin,elevation").split(",") if x.strip()]

    _here = os.path.dirname(os.path.abspath(__file__))
    ckpt = CKPT if os.path.isabs(CKPT) else os.path.join(_here, CKPT)
    path_file = PATH_FILE if os.path.isabs(PATH_FILE) else os.path.join(_here, PATH_FILE)

    print(f"Loading MarioLM: {ckpt}")
    mario_lm = MarioLM(lm_path=ckpt, tokenizer_path=TOK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()
    sampler = SBSSampler(mario_lm, temperature=TEMP, top_k=TOPK)

    path_grid = load_path_grid(path_file, H)
    print(f"path: {path_file}  (W={len(path_grid[0])})")

    # 固定一個 target_prompt（跨 greedy/search/trials 共用），用 prompter 抽以保證字詞/模板一致
    target_prompt = os.environ.get("TARGET_PROMPT")
    if not target_prompt:
        target_prompt = mario_lm.prompter(sample_prompt=True)[0]
    print(f"target_prompt: {target_prompt}")

    # 固定條件嵌入，避免每次 env 重算
    with torch.no_grad():
        ehs = mario_lm.prompter.output_hidden(target_prompt).view(1, 1, -1).to(device)

    def make_env():
        return LevelGenEnv(
            sampler=sampler, path_grid=path_grid, target_prompt=target_prompt,
            encoder_hidden_states=ehs, seg_cols=SEG_COLS, target_cols=TARGET_COLS,
            H=H, K=K, lambda_s=LAMBDA_S, lambda_p=LAMBDA_P,
            reward_features=REWARD_FEATURES, use_mock_astar=True,
            temperature=TEMP,
        )

    print(f"\nTARGET_COLS={TARGET_COLS} SEG_COLS={SEG_COLS} K={K} "
          f"N_TRIALS={N_TRIALS} temp={TEMP} top_k={TOPK}\n")

    g_rew, s_rew = [], []
    g_struct, s_struct = [], []
    g_path, s_path = [], []
    g_solv, s_solv = [], []
    g_cost, s_cost = [], []

    for t in range(N_TRIALS):
        # greedy
        env = make_env()
        ctr = ProposeCounter(env)
        rd = rollout_greedy(env, ctr, env.reset())
        g_rew.append(rd["reward"]); g_struct.append(rd["adh_struct"])
        g_path.append(rd["adh_path"]); g_solv.append(rd["solvable"]); g_cost.append(ctr.candidates)

        # search
        env = make_env()
        ctr = ProposeCounter(env)
        rd = search_lookahead(env, ctr, env.reset())
        s_rew.append(rd["reward"]); s_struct.append(rd["adh_struct"])
        s_path.append(rd["adh_path"]); s_solv.append(rd["solvable"]); s_cost.append(ctr.candidates)

        print(f"  trial {t+1:2d}: greedy={g_rew[-1]:.3f}  search={s_rew[-1]:.3f}  "
              f"gain={s_rew[-1]-g_rew[-1]:+.3f}")

    def m(xs):
        return st.mean(xs)

    print("\n" + "=" * 56)
    print(f"{'metric':14s} {'greedy':>10s} {'search':>10s} {'Δ':>10s}")
    print("-" * 56)
    print(f"{'reward':14s} {m(g_rew):10.3f} {m(s_rew):10.3f} {m(s_rew)-m(g_rew):+10.3f}")
    print(f"{'adh_struct':14s} {m(g_struct):10.3f} {m(s_struct):10.3f} {m(s_struct)-m(g_struct):+10.3f}")
    print(f"{'adh_path':14s} {m(g_path):10.3f} {m(s_path):10.3f} {m(s_path)-m(g_path):+10.3f}")
    print(f"{'solvable':14s} {m(g_solv):10.3f} {m(s_solv):10.3f} {m(s_solv)-m(g_solv):+10.3f}")
    print(f"{'#candidates':14s} {m(g_cost):10.1f} {m(s_cost):10.1f} {m(s_cost)-m(g_cost):+10.1f}")
    print("=" * 56)
    gain = m(s_rew) - m(g_rew)
    print(f"\nsearch - greedy reward gain = {gain:+.3f}"
          f"   ({'search 贏' if gain > 0 else 'search 未贏，需調 K/seg/path 難度'})")


if __name__ == "__main__":
    main()