"""
Stochastic Beam Search (SBS) proposal for dual-expert MarioGPT
—— 對齊 ChenFrite/duo_mariogpt structure-expert 的 sampler.py 實際結構。

與原始 MarioGPT 的差異（已對齊你的程式）：
  1. 生成單位是「欄」(column, H=14 tile)，逐 token 但語意單位是欄。
  2. path 以 prefix (target+future) + forced_ids 硬約束注入，不是 cross-attention。
  3. position_2d 是逐 token 的 (x, y) 座標對。
  4. 目前無 MoE → log-prob 為標準序列 log-prob，無需 router 邊際化。

SBS 提案的一個 action = 連續 seg_cols 欄的 structural segment。
forced（path token）位置不採樣、直接填 path token 且不計入 Gumbel 擾動的自由度，
只有「非 forced 位置」參與 without-replacement 的 top-k。

用法：把 SBSMixin 混入 GPTSampler，或把 propose_segment 貼進 GPTSampler。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _gumbel(shape, device, dtype, eps: float = 1e-20) -> torch.Tensor:
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def _shifted_gumbel(logprob: torch.Tensor, parent_G: torch.Tensor) -> torch.Tensor:
    """
    SBS 核心：conditional / shifted Gumbel（Kool et al. 2019）。
    保證 max_i G_tilde_i == parent_G，使整段序列的 top-k
    等於「從完整序列分布 without replacement 抽 k 個」。
    這是 proposal Proposition 1 成立的實作前提。
    """
    G = logprob + _gumbel(logprob.shape, logprob.device, logprob.dtype)
    Z = G.max()

    def log1mexp(x):  # log(1 - exp(x))，x <= 0
        return torch.where(
            x > -0.6931,
            torch.log(-torch.expm1(x.clamp(max=-1e-7))),
            torch.log1p(-torch.exp(x)),
        )

    v = parent_G - G + log1mexp(G - Z)
    return parent_G - F.relu(v) - torch.log1p(torch.exp(-torch.abs(v)))


@dataclass
class SegmentCandidate:
    """一個 sampled-action：連續 seg_cols 欄的 structural segment。"""
    tile_columns: List[List[int]]   # [seg_cols][H]  每欄 H 個 token id
    logprob: float                  # log π_θ(a | s_t)，序列層級（僅非 forced 位置累積）
    perturbed: float                # 最終 shifted-Gumbel 值（排序/除錯）


class SBSMixin:
    """
    宿主（GPTSampler）需提供：
      self.mario_lm.lm, self.mario_lm.tokenizer, self.device
      self.temperature, self.top_k, self.logits_warper
      self._encode_char, self._next_token（本檔改用自己的 _step_logprob）
    以及 sampler 內的 PATH_TOKENS / H 概念。
    """

    @torch.no_grad()
    def _step_logprob(
        self,
        input_ids: torch.Tensor,       # [1, L]
        position_2d: torch.Tensor,     # [1, L, 2]
        encoder_hidden_states: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """回傳當前步 next-token 的 log π_θ（[vocab]）。無 MoE → 標準 log_softmax。"""
        attn = torch.ones_like(input_ids)
        out = self.mario_lm.lm(
            input_ids=input_ids,
            attention_mask=attn,
            encoder_hidden_states=encoder_hidden_states,
            position_2d=position_2d,
        )
        logits = out.logits[0, -1, :] / temperature
        # <|endoftext|>（id 0）decode 成 13 字元而非 1 個 tile，會讓 level row 長度不一 → 禁止採樣
        eos_id = self.mario_lm.tokenizer.eos_token_id
        if eos_id is not None:
            logits[eos_id] = float("-inf")
        return F.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def propose_segment(
        self,
        prefix_ids: List[int],          # target_path + future_path + struct_context（同 _sample_patch）
        prefix_xs: List[float],
        prefix_ys: List[int],
        struct_xs: List[float],         # 這一段要生成位置的 x（長度 seg_cols*H）
        struct_ys: List[int],
        encoder_hidden_states: torch.Tensor,
        forced_ids: List[Optional[int]],  # 長度 seg_cols*H；path token 位置為 int，其餘 None
        H: int,
        seg_cols: int,
        K: int,
        PATH_TOKENS_ENCODED: Optional[frozenset] = None,  # 可選：加速判斷
        temperature: Optional[float] = None,
    ) -> List[SegmentCandidate]:
        """
        以 SBS 抽 K 個「下一段」(seg_cols 欄) 候選。
        forced 位置直接填 path token（不採樣、不擾動）；
        非 forced 位置參與序列層級 Gumbel-Top-k without replacement。

        回傳 K 個 SegmentCandidate，作為 sampled-action MDP 的節點局部動作表 A(s_t)。
        """
        temp = temperature if temperature is not None else self.temperature
        device = self.device
        seg_len = seg_cols * H
        assert len(struct_xs) == seg_len and len(forced_ids) == seg_len

        # K 條 beam：每條存 已生成 token、累積 logprob、shifted-Gumbel
        beam_gen: List[List[int]] = [[]]
        beam_logprob = torch.zeros(1, device=device)
        beam_G = _gumbel((1,), device, beam_logprob.dtype)

        for step in range(seg_len):
            forced_tok = forced_ids[step]

            if forced_tok is not None:
                # forced（path token）位置：所有 beam 直接 append，不採樣不擾動
                for b in range(len(beam_gen)):
                    beam_gen[b].append(forced_tok)
                # logprob / G 不變（forced 不是模型的自由選擇）
                continue

            # 非 forced：對每條 beam 算 next-token log-prob，展開子節點
            cand_G, cand_logprob, cand_beam, cand_tok = [], [], [], []
            for b in range(len(beam_gen)):
                cur_ids = prefix_ids + beam_gen[b]
                input_ids = torch.tensor([cur_ids], dtype=torch.long, device=device)

                xs = prefix_xs + struct_xs[:step]
                ys = prefix_ys + struct_ys[:step]
                pos = torch.tensor(
                    list(zip(xs, ys)), dtype=torch.float32, device=device
                ).unsqueeze(0)

                logp = self._step_logprob(input_ids, pos, encoder_hidden_states, temp)  # [vocab]

                # 可選：只在 top_k 個 token 上展開，控制分支數（對齊你 sampler 的 top_k）
                if getattr(self, "top_k", 0) and self.top_k > 0:
                    topv, topi = torch.topk(logp, min(self.top_k, logp.shape[0]))
                    child_lp = beam_logprob[b] + topv
                    child_G = _shifted_gumbel(child_lp, beam_G[b])
                    cand_G.append(child_G)
                    cand_logprob.append(child_lp)
                    cand_beam.append(torch.full_like(child_lp, b, dtype=torch.long))
                    cand_tok.append(topi)
                else:
                    child_lp = beam_logprob[b] + logp
                    child_G = _shifted_gumbel(child_lp, beam_G[b])
                    cand_G.append(child_G)
                    cand_logprob.append(child_lp)
                    cand_beam.append(torch.full_like(child_lp, b, dtype=torch.long))
                    cand_tok.append(torch.arange(logp.shape[0], device=device))

            cand_G = torch.cat(cand_G)
            cand_logprob = torch.cat(cand_logprob)
            cand_beam = torch.cat(cand_beam)
            cand_tok = torch.cat(cand_tok)

            topk = min(K, cand_G.shape[0])
            sel = torch.topk(cand_G, topk).indices

            new_gen, new_lp, new_G = [], [], []
            for idx in sel:
                b = int(cand_beam[idx])
                new_gen.append(beam_gen[b] + [int(cand_tok[idx])])
                new_lp.append(cand_logprob[idx])
                new_G.append(cand_G[idx])
            beam_gen = new_gen
            beam_logprob = torch.stack(new_lp)
            beam_G = torch.stack(new_G)

        # 把每條 beam 的 seg_len token 切成 seg_cols 欄
        candidates: List[SegmentCandidate] = []
        for i in range(len(beam_gen)):
            toks = beam_gen[i]
            cols = [toks[c * H:(c + 1) * H] for c in range(seg_cols)]
            candidates.append(
                SegmentCandidate(
                    tile_columns=cols,
                    logprob=float(beam_logprob[i]),
                    perturbed=float(beam_G[i]),
                )
            )
        candidates.sort(key=lambda c: c.logprob, reverse=True)
        return candidates
