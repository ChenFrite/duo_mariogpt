"""
跑通「propose K 個候選」的最小測試腳本。

放在 structure-expert/mario_gpt_structure/ 目錄下執行（與 sampler.py 同層），
或調整 import 路徑。它會：
  1. 載入你的雙專家 MarioGPT
  2. 把 SBSMixin 混入 GPTSampler
  3. 用一個 path_grid 跑一次 propose_segment，抽 K 個候選
  4. 驗證：(a) 候選在「非 forced 位置」不重複；(b) logprob 與 teacher-forcing 一致

先確認這一步對了，再往 LevelGenEnv / 搜索推進。
"""

from __future__ import annotations

import os
import sys

from typing import List, Optional
import torch
import torch.nn.functional as F

# 直接从同目录的 sampler.py import（不走 mario_gpt_structure.sampler 套件）
from mario_gpt_structure.sampler import (
    GPTSampler,
    PATH_TOKENS,
    PATCH_WIDTH,
    FUTURE_WIDTH,
    _patch_positions,
    load_path_grid,
)

from mario_gpt_structure import MarioLM
from mario_gpt_structure.sbs_propose import SBSMixin, SegmentCandidate

# ============================================================
# 1. 混入：讓 GPTSampler 具備 propose_segment
# ============================================================
class SBSSampler(SBSMixin, GPTSampler):
    """GPTSampler + SBS 提案能力。其餘行為與原 GPTSampler 相同。"""
    pass


# ============================================================
# 2. 準備一次 propose 的輸入（複用 sampler 內部的建構方式）
# ============================================================
def build_propose_inputs(
    sampler: SBSSampler,
    path_grid: List[List[str]],
    H: int,
    seg_cols: int,
    encoder_hidden_states: torch.Tensor,
):
    """
    模擬 sampler.sample 裡「block 0」的 prefix 建構，
    但把要生成的欄數改成 seg_cols（一個 SBS action 的粒度）。
    回傳 propose_segment 需要的所有參數。
    """
    device = sampler.device
    path_start = 0
    new_start = 0  # 從頭開始生成一段

    # target pure path: PATCH_WIDTH 欄
    target_ids: List[int] = []
    for c_local in range(PATCH_WIDTH):
        c_global = path_start + c_local
        for r in range(H):
            target_ids.append(sampler._encode_char(path_grid[r][c_global]))

    # future pure path: FUTURE_WIDTH 欄
    future_ids: List[int] = []
    for c_local in range(FUTURE_WIDTH):
        c_global = path_start + PATCH_WIDTH + c_local
        for r in range(H):
            future_ids.append(sampler._encode_char(path_grid[r][c_global]))

    target_xs, target_ys = _patch_positions(H, PATCH_WIDTH, x_offset=0)
    future_xs, future_ys = _patch_positions(H, FUTURE_WIDTH, x_offset=PATCH_WIDTH)

    # block 0 無 struct context
    prefix_ids = target_ids + future_ids
    prefix_xs = target_xs + future_xs
    prefix_ys = target_ys + future_ys

    # 要生成的 seg_cols 欄的位置（local x = 0..seg_cols-1）
    struct_xs: List[float] = []
    struct_ys: List[int] = []
    for i in range(seg_cols):
        x = float((new_start + i) - path_start)
        struct_xs += [x] * H
        struct_ys += list(range(H))

    # forced_ids：path token 位置強制填
    forced_ids: List[Optional[int]] = [None] * (H * seg_cols)
    W = len(path_grid[0])
    for i in range(seg_cols):
        g_col = new_start + i
        if g_col < W:
            for r in range(H):
                ch = path_grid[r][g_col]
                if ch in PATH_TOKENS:
                    forced_ids[i * H + r] = sampler._encode_char(ch)

    return dict(
        prefix_ids=prefix_ids,
        prefix_xs=prefix_xs,
        prefix_ys=prefix_ys,
        struct_xs=struct_xs,
        struct_ys=struct_ys,
        encoder_hidden_states=encoder_hidden_states,
        forced_ids=forced_ids,
        H=H,
        seg_cols=seg_cols,
    )


# ============================================================
# 3. 驗證：teacher-forcing 重算某候選的 log-prob（僅非 forced 位置）
# ============================================================
@torch.no_grad()
def teacher_forcing_logprob(
    sampler: SBSSampler,
    prefix_ids: List[int],
    prefix_xs: List[float],
    prefix_ys: List[int],
    struct_xs: List[float],
    struct_ys: List[int],
    forced_ids: List[Optional[int]],
    candidate: SegmentCandidate,
    encoder_hidden_states: torch.Tensor,
    H: int,
    temperature: float,
) -> float:
    """把 candidate 的 token 逐步餵回模型，累積『非 forced 位置』的 log-prob。"""
    device = sampler.device
    # 攤平候選的 token（seg_cols 欄 → seg_len token）
    cand_tokens: List[int] = []
    for col in candidate.tile_columns:
        cand_tokens.extend(col)

    total_lp = 0.0
    generated: List[int] = []
    for step, tok in enumerate(cand_tokens):
        if forced_ids[step] is not None:
            # forced 位置不計入 log-prob（與 propose 一致）
            generated.append(tok)
            continue

        cur_ids = prefix_ids + generated
        input_ids = torch.tensor([cur_ids], dtype=torch.long, device=device)
        xs = prefix_xs + struct_xs[:step]
        ys = prefix_ys + struct_ys[:step]
        pos = torch.tensor(list(zip(xs, ys)), dtype=torch.float32, device=device).unsqueeze(0)

        logp = sampler._step_logprob(input_ids, pos, encoder_hidden_states, temperature)
        total_lp += float(logp[tok])
        generated.append(tok)

    return total_lp


# ============================================================
# 4. 主測試
# ============================================================
def main():
    H = 14
    seg_cols = 2         # 一個 action = 2 欄（先小，驗證用）
    K = 4
    temperature = 2.0

    # ---- 載入模型（改成你的 checkpoint 路徑）----
    CKPT = os.environ.get("MARIO_CKPT", "Mario-GPT2-700-context-length")
    print(f"Loading model from {CKPT} ...")
    mario_lm = MarioLM(lm_path=CKPT, tokenizer_path=CKPT)
    mario_lm.eval()

    sampler = SBSSampler(mario_lm, temperature=temperature, top_k=16)

    # ---- 準備 path_grid：至少要 PATCH_WIDTH + FUTURE_WIDTH 欄寬 ----
    PATH_FILE = os.environ.get("PATH_FILE", "")
    if PATH_FILE and os.path.exists(PATH_FILE):
        path_grid = load_path_grid(PATH_FILE, H)
        print(f"Loaded path from {PATH_FILE}, width={len(path_grid[0])}")
    else:
        # 沒有 path 檔就用全 '-'（無 path 約束，純測 SBS 提案）
        W = PATCH_WIDTH + FUTURE_WIDTH + seg_cols
        path_grid = [['-'] * W for _ in range(H)]
        print(f"No path file; using empty path_grid width={W}")

    # ---- encoder_hidden_states（prompt 條件）----
    with torch.no_grad():
        enc = mario_lm.prompter(sample_prompt=True)[1]
        encoder_hidden_states = enc.view(1, 1, -1).to(sampler.device)

    inputs = build_propose_inputs(sampler, path_grid, H, seg_cols, encoder_hidden_states)

    # ---- 跑 propose ----
    print(f"\nRunning propose_segment: K={K}, seg_cols={seg_cols} ...")
    candidates = sampler.propose_segment(K=K, temperature=temperature, **inputs)
    print(f"Got {len(candidates)} candidates.\n")

    for i, c in enumerate(candidates):
        # 解碼成可讀的 tile（每欄 H token）
        cols_str = []
        for col in c.tile_columns:
            chars = "".join(mario_lm.tokenizer.decode([t]) for t in col)
            cols_str.append(chars)
        print(f"[cand {i}] logprob={c.logprob:.3f}  perturbed={c.perturbed:.3f}")
        print(f"          cols={cols_str}")

    # ================= 測試 1：非 forced 位置不重複 =================
    forced = inputs["forced_ids"]
    seg_len = seg_cols * H

    def free_signature(cand: SegmentCandidate):
        toks = [t for col in cand.tile_columns for t in col]
        return tuple(toks[s] for s in range(seg_len) if forced[s] is None)

    sigs = [free_signature(c) for c in candidates]
    n_unique = len(set(sigs))
    n_free = sum(1 for f in forced if f is None)
    print(f"\n[Test 1] free positions per candidate = {n_free}")
    print(f"[Test 1] unique candidates (by free positions) = {n_unique}/{len(candidates)}")
    if n_free == 0:
        print("[Test 1] path 太密（全 forced），SBS 無自由度 → 候選必然相同（非 bug）")
    elif n_unique == len(candidates):
        print("[Test 1] PASS：候選兩兩不同（without-replacement 生效）")
    else:
        print("[Test 1] WARN：有重複候選。若 free 位置多仍重複 → shifted Gumbel 可能有誤")

    # ================= 測試 2：logprob 與 teacher-forcing 一致 =================
    print("\n[Test 2] teacher-forcing 重算 logprob（僅非 forced 位置）...")
    for i, c in enumerate(candidates):
        tf_lp = teacher_forcing_logprob(
            sampler,
            inputs["prefix_ids"], inputs["prefix_xs"], inputs["prefix_ys"],
            inputs["struct_xs"], inputs["struct_ys"], forced,
            c, encoder_hidden_states, H, temperature,
        )
        diff = abs(tf_lp - c.logprob)
        status = "PASS" if diff < 1e-3 else "FAIL"
        print(f"  [cand {i}] propose={c.logprob:.4f}  tf={tf_lp:.4f}  |diff|={diff:.2e}  {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
