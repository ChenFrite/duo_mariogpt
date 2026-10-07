from __future__ import annotations

import os, sys

from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from mario_gpt_structure.level import FULL_LEVEL_STR_WITH_PATHS

PATH_TOKENS = frozenset({
    'M', 'J', 'j', 'A', 'C',   # 原始路徑TOKEN
    'L',                         # 滯空的M
    'K', 'W', 'U',               # 大寫J衍生：滯空/終點/A上方
    'k', 'w',                    # 小寫j衍生：滯空/終點
    'R', 'r',                    # 右牆弧線TOKEN
})
PIPE_RIGHT  = frozenset({'>', ']'})  # orphaned at patch left edge  → replace with X
PIPE_LEFT   = frozenset({'<', '['})  # orphaned at patch right edge → replace with X
DEFAULT_MODEL = "distilgpt2"
PATCH_WIDTH = 25   # prediction target width (columns)
FUTURE_WIDTH = 10  # future context width appended after target path
PATCH_STRIDE = 1   # sliding window stride


def _encode_char(tokenizer, ch: str) -> int:
    return tokenizer.encode(ch, add_special_tokens=False)[0]


class MarioDataset(Dataset):
    """
    Patch-based dataset: pure-path prompt + full-structure target.

    The level is sliced into overlapping 14×PATCH_WIDTH patches using a
    sliding window of stride PATCH_STRIDE (1 = maximum overlap: patch
    starts are 0,1,2,3,... i.e. columns 0-24, 1-25, 2-26, ...). For each
    patch, the training sequence is:

        [ pure_path(H*PATCH_WIDTH) | structure(H*PATCH_WIDTH) ]

    Both halves are flattened column-major (col0 row0..H-1, col1 row0..H-1, ...).
    pure_path replaces every non-PATH_TOKEN character with '-'.
    structure is the complete tile map including path tokens at their positions.

    Loss is computed on all structure positions (path tokens and non-path tiles alike).

    2D absolute positions (x = column index within patch, shared by both
    halves so a structure token aligns with its corresponding path token):
        pure_path  col c_local: x=c_local, y=0..H-1
        structure  col c_local: x=c_local, y=0..H-1

    __getitem__ returns (input_ids, loss_mask, positions, attn_mask).
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        level_string: Optional[str] = None,
        height: int = 14,
    ):
        if level_string is None:
            level_string = FULL_LEVEL_STR_WITH_PATHS

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.tokenizer = tokenizer
        self.height = height
        H = height
        self.patch_width = PATCH_WIDTH
        self.future_width = FUTURE_WIDTH
        self.seq_len = (2 * PATCH_WIDTH + FUTURE_WIDTH) * H  # 350+140+350 = 840

        rows = [r for r in level_string.split("\n") if r.strip()]
        assert len(rows) == H, f"Expected {H} rows, got {len(rows)}"
        W = len(rows[0])

        # tile_cols[c][r] = actual tile char at col c, row r (global index)
        tile_cols: List[List[str]] = []
        for c in range(W):
            t_col = [rows[r][c] if c < len(rows[r]) else '-' for r in range(H)]
            tile_cols.append(t_col)


        samples_ids:  List[torch.Tensor] = []
        samples_mask: List[torch.Tensor] = []
        samples_pos:  List[torch.Tensor] = []
        samples_attn: List[torch.Tensor] = []

        num_patches = 0
        for patch_start in range(0, W - PATCH_WIDTH - FUTURE_WIDTH + 1, PATCH_STRIDE):
            num_patches += 1

            pure_path_ids:   List[int] = []
            future_path_ids: List[int] = []
            structure_ids:   List[int] = []
            xs: List[float] = []
            ys: List[int] = []

            # --- target pure path (no loss), x=0~24 ---
            for c_local in range(PATCH_WIDTH):
                c_global = patch_start + c_local
                for r in range(H):
                    ch = tile_cols[c_global][r]
                    path_ch = ch if ch in PATH_TOKENS else '-'
                    pure_path_ids.append(_encode_char(tokenizer, path_ch))
                xs += [float(c_local)] * H
                ys += list(range(H))

            # --- future pure path (no loss), x=25~34 ---
            for c_local in range(FUTURE_WIDTH):
                c_global = patch_start + PATCH_WIDTH + c_local
                for r in range(H):
                    ch = tile_cols[c_global][r]
                    path_ch = ch if ch in PATH_TOKENS else '-'
                    future_path_ids.append(_encode_char(tokenizer, path_ch))
                xs += [float(PATCH_WIDTH + c_local)] * H
                ys += list(range(H))

            # --- full structure with path tokens (ALL positions have loss), x=0~24 ---
            struct_loss: List[int] = []
            for c_local in range(PATCH_WIDTH):  # x=0..24 (25 cols, 350 tokens)
                c_global = patch_start + c_local
                is_first = (c_local == 0)
                is_last  = (c_local == PATCH_WIDTH - 1)
                for r in range(H):
                    ch = tile_cols[c_global][r]
                    if is_first and ch in PIPE_RIGHT:
                        ch = 'X'
                    elif is_last and ch in PIPE_LEFT:
                        ch = 'X'
                    structure_ids.append(_encode_char(tokenizer, ch))
                    struct_loss.append(1)  # all positions have loss
                xs += [float(c_local)] * H
                ys += list(range(H))

            input_ids = pure_path_ids + future_path_ids + structure_ids
            loss_mask = [0] * ((PATCH_WIDTH + FUTURE_WIDTH) * H) + struct_loss
            attn_mask = [1] * len(input_ids)

            positions = torch.tensor(list(zip(xs, ys)), dtype=torch.float32)

            samples_ids.append(torch.tensor(input_ids, dtype=torch.long))
            samples_mask.append(torch.tensor(loss_mask, dtype=torch.long))
            samples_pos.append(positions)
            samples_attn.append(torch.tensor(attn_mask, dtype=torch.long))

        self.samples_ids  = samples_ids
        self.samples_mask = samples_mask
        self.samples_pos  = samples_pos
        self.samples_attn = samples_attn

        self.character_set = set(ch for col in tile_cols for ch in col)
        print(
            f"MarioDataset: {W} cols, {num_patches} patches (PATCH_WIDTH={PATCH_WIDTH}, "
            f"FUTURE_WIDTH={FUTURE_WIDTH}, PATCH_STRIDE={PATCH_STRIDE}), "
            f"H={H}, SEQ_LEN={self.seq_len}, samples={len(samples_ids)}, chars={self.character_set}"
        )

    def __len__(self) -> int:
        return len(self.samples_ids)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return (
                self.samples_ids[idx],
                self.samples_mask[idx],
                self.samples_pos[idx],
                self.samples_attn[idx],
            )

        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        idx = [int(i) for i in idx]

        input_ids = torch.stack([self.samples_ids[i]  for i in idx])
        loss_mask = torch.stack([self.samples_mask[i] for i in idx])
        positions = torch.stack([self.samples_pos[i]  for i in idx])
        attn_mask = torch.stack([self.samples_attn[i] for i in idx])
        return input_ids, loss_mask, positions, attn_mask
