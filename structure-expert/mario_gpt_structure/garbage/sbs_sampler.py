"""
Stochastic Beam Search (SBS) for dual-expert MarioGPT.

依 Kool, van Hoof, Welling (ICML 2019) "Stochastic Beams and Where to Find Them"
序列層級 Gumbel-Top-k without replacement，回傳 K 個下一段候選 + log π_θ（含 MoE router 邊際化）。

設計對齊 gumbel-alphazero-mariogpt-proposal：
  {(a_i, log π_θ(a_i | s_t))}_{i=1..K}  ←  SBS(G, s_t, K)
每個 a_i 是變長 structural segment（column-chunk），供 sampled-action MDP 當節點局部動作表。

把本檔的 SBSMixin 混入現有 GPTSampler，或把 propose() 直接貼進 GPTSampler。
不改動原有 __call__ 生成流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Gumbel 工具
# ----------------------------------------------------------------------
def _gumbel(shape, device, dtype, eps: float = 1e-20) -> torch.Tensor:
    """標準 Gumbel(0,1) 抽樣。"""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + eps) + eps)


def _shifted_gumbel(logprob: torch.Tensor, parent_G: torch.Tensor) -> torch.Tensor:
    """
    SBS 的核心：conditional / shifted Gumbel。

    給定某父節點的擾動值 parent_G（= 父序列 logprob 的 Gumbel-top 值），
    以及該父節點展開出的所有子節點的累積 logprob（logprob，shape [n_children]），
    回傳「已 shift」的子節點 Gumbel 值 G_tilde，使得
        max_i G_tilde_i == parent_G
    這保證整棵樹上每一層的 top-k 邊際分布，等於「從完整序列分布 without replacement 抽 k 個」。

    公式（Kool et al. 2019, Appendix B, numerically stable 版本）：
        G_i        = logprob_i + Gumbel_i          # 原始擾動
        Z          = max_i G_i
        v_i        = parent_G - G_i + log1mexp(G_i - Z)
        G_tilde_i  = parent_G - relu(v_i) - log1pexp(-|v_i|)
    """
    G = logprob + _gumbel(logprob.shape, logprob.device, logprob.dtype)
    Z = G.max()

    # log(1 - exp(x)) for x <= 0，數值穩定
    def log1mexp(x):
        # x = G_i - Z <= 0
        return torch.where(
            x > -0.6931,  # -ln2
            torch.log(-torch.expm1(x.clamp(max=-1e-7))),
            torch.log1p(-torch.exp(x)),
        )

    v = parent_G - G + log1mexp(G - Z)
    # G_tilde = parent_G - softplus_like，保證 max == parent_G
    G_tilde = parent_G - F.relu(v) - torch.log1p(torch.exp(-torch.abs(v)))
    return G_tilde


# ----------------------------------------------------------------------
# SBS 候選容器
# ----------------------------------------------------------------------
@dataclass
class SBSCandidate:
    tokens: torch.Tensor        # [seg_len]  該段 structural segment 的 token ids
    logprob: float              # log π_θ(a_i | s_t)，序列層級，含 router 邊際化
    perturbed: float            # 該候選最終的 shifted-Gumbel 值（排序用，除錯用）


# ----------------------------------------------------------------------
# SBS mixin：混入 GPTSampler
# ----------------------------------------------------------------------
class SBSMixin:
    """
    需要宿主（GPTSampler）提供：
      self.mario_lm.lm         : 帶 cross-attention 的 GPT2（雙專家）
      self.mario_lm.tokenizer  : tokenizer
      self.temperature         : float
      self.context_len         : int
    以及每欄 token 數（Mario 14 列 → 每欄 14 個 tile token）。
    """

    # Mario 一欄 = level_height 個 tile token；一段 = segment_cols 欄
    SEG_COLS_DEFAULT = 4          # 一個 action 提案幾欄（proposal 的 chunk 粒度）
    LEVEL_HEIGHT_DEFAULT = 14

    @torch.no_grad()
    def propose(
        self,
        prefix_ids: torch.Tensor,                 # [prefix_len] 目前前綴 ℓ_1:t（單一 batch）
        K: int,
        encoder_hidden_states: torch.Tensor,      # 條件 c 的 cross-attention memory（固定）
        segment_cols: Optional[int] = None,
        level_height: Optional[int] = None,
        positions_2d: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> List[SBSCandidate]:
        """
        從雙專家 G 以 SBS 抽 K 個「下一段」候選。
        回傳 K 個 SBSCandidate，logprob 已含 MoE router 邊際化（因為 lm.forward
        的 logits 是對 router 加權後的輸出，softmax 後即邊際化機率）。

        s_t = (prefix_ids, encoder_hidden_states)，動力學確定：append 後即 s_{t+1}。
        """
        seg_cols = segment_cols or self.SEG_COLS_DEFAULT
        H = level_height or self.LEVEL_HEIGHT_DEFAULT
        seg_len = seg_cols * H                      # 這一段要生成的 token 數
        temp = temperature if temperature is not None else self.temperature
        device = prefix_ids.device
        model = self.mario_lm.lm

        # ---- beam 狀態：K 條 beam 同步往前展開 seg_len 步 ----
        # 每條 beam 存：已生成的 segment token、累積 logprob、shifted-Gumbel 值
        beam_tokens = [torch.empty(0, dtype=torch.long, device=device) for _ in range(1)]
        beam_logprob = torch.zeros(1, device=device)
        # 根：parent_G = 標準 Gumbel（單一根，logprob=0）
        beam_G = _gumbel((1,), device, beam_logprob.dtype)

        for step in range(seg_len):
            # 對每條 beam，前綴 = prefix_ids ⊕ 該 beam 已生成的 segment
            all_next_logprob = []   # 每條 beam 的 next-token logprob 分布
            for b in range(len(beam_tokens)):
                full_ids = torch.cat([prefix_ids, beam_tokens[b]], dim=0)
                # 只取最後 context_len 個 token（對齊原始 sampler 的 window）
                inp = full_ids[-self.context_len :].unsqueeze(0)  # [1, L]

                # 2D position（若模型需要）— 對齊你 sampler 的 positions_2d 用法
                model_kwargs = dict(encoder_hidden_states=encoder_hidden_states)
                if positions_2d is not None:
                    # 取對應 window 的 position 切片；實作依你的 GPT2With2DSinusoids 介面
                    model_kwargs["position_2d"] = positions_2d[:, : inp.shape[1], :]

                out = model(input_ids=inp, **model_kwargs)
                logits = out.logits[0, -1, :] / temp          # [vocab]
                # 這裡的 logits 已是 MoE router 加權後輸出 → softmax 即 router 邊際化
                logp = F.log_softmax(logits, dim=-1)          # log π_θ(next | ...)
                all_next_logprob.append(logp)

            # ---- 序列層級 Gumbel-Top-k without replacement ----
            # 對每條 beam，展開所有 vocab 子節點：子 logprob = beam_logprob + logp
            # 子節點的 shifted Gumbel = _shifted_gumbel(child_logprob, parent_G=beam_G[b])
            cand_tok = []
            cand_beam = []
            cand_logprob = []
            cand_G = []
            for b in range(len(beam_tokens)):
                child_logprob = beam_logprob[b] + all_next_logprob[b]     # [vocab]
                child_G = _shifted_gumbel(child_logprob, beam_G[b])        # [vocab]
                cand_logprob.append(child_logprob)
                cand_G.append(child_G)
                cand_beam.append(torch.full_like(child_logprob, b, dtype=torch.long))
                cand_tok.append(torch.arange(child_logprob.shape[0], device=device))

            cand_logprob = torch.cat(cand_logprob)   # [n_beam * vocab]
            cand_G = torch.cat(cand_G)
            cand_beam = torch.cat(cand_beam)
            cand_tok = torch.cat(cand_tok)

            # 取 shifted-Gumbel 最大的 K 個（= without-replacement 的 top-k）
            topk = min(K, cand_G.shape[0])
            sel = torch.topk(cand_G, topk).indices

            new_tokens, new_logprob, new_G = [], [], []
            for idx in sel:
                b = int(cand_beam[idx])
                tok = cand_tok[idx].view(1)
                new_tokens.append(torch.cat([beam_tokens[b], tok], dim=0))
                new_logprob.append(cand_logprob[idx])
                new_G.append(cand_G[idx])

            beam_tokens = new_tokens
            beam_logprob = torch.stack(new_logprob)
            beam_G = torch.stack(new_G)

        # ---- seg_len 步後，K 條 beam 即 K 個 segment 候選 ----
        candidates = [
            SBSCandidate(
                tokens=beam_tokens[i],
                logprob=float(beam_logprob[i]),
                perturbed=float(beam_G[i]),
            )
            for i in range(len(beam_tokens))
        ]
        # 依 logprob 由高到低回傳（節點局部動作表 index 0..K-1）
        candidates.sort(key=lambda c: c.logprob, reverse=True)
        return candidates
