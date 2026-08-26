from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from PIL.Image import Image
from tqdm import tqdm
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.utils import convert_level_to_png, load_level, save_level

PATH_TOKENS = frozenset({
    'M', 'J', 'j', 'A', 'C',   # 原始路徑TOKEN
    'L',                         # 滯空的M
    'K', 'W', 'U',               # 大寫J衍生：滯空/終點/A上方
    'k', 'w',                    # 小寫j衍生：滯空/終點
    'R', 'r',                    # 右牆弧線TOKEN
})
PATCH_WIDTH   = 25  # must match dataset.py
FUTURE_WIDTH  = 10  # must match dataset.py
BLOCK_NEW     = 1                       # new cols per sliding block (k>=1)
BLOCK_CONTEXT = PATCH_WIDTH - BLOCK_NEW  # 3 overlap cols for blocks k>=1
# Block 0: full 25-col AR (x=0..24), no context → matches training seq exactly (840 tok)
# Block 1+: 3-22 sliding window (3 col context at x=0..2, generate 22 new cols at x=3..24)


def load_path_grid(path_file: str, height: int = 14) -> List[List[str]]:
    """
    Load pure path map from file. Returns path_grid[row][col].
    File format: path tokens as-is, '-' above topmost path token, 'P' below.
    """
    with open(path_file, "r") as f:
        lines = f.read().splitlines()
    rows = [ln for ln in lines if ln]
    assert len(rows) == height, f"Expected {height} rows, got {len(rows)}"
    W = len(rows[0])
    grid: List[List[str]] = []
    for r in range(height):
        row_chars = [
            rows[r][c] if c < len(rows[r]) else '-'
            for c in range(W)
        ]
        grid.append(row_chars)
    return grid  # [H][W]


def load_forced_path_from_file(path_file: str, height: int = 14, tokenizer=None):
    return load_path_grid(path_file, height)


@dataclass
class SampleOutput:
    level: Optional[List[str]]
    prompt: Optional[str] = None
    img: Optional[Image] = None
    level_tensor: Optional[torch.Tensor] = None

    @classmethod
    def from_rows(cls, level_rows: List[str], tokenizer, prompter=None) -> "SampleOutput":
        img = None
        try:
            img = convert_level_to_png(level_rows)[0]
        except Exception as e:
            print(f"Image render failed: {e}")
        return cls(level=level_rows, img=img)

    def save(self, filename: str) -> str:
        if self.level is None:
            raise ValueError("No level data to save.")
        with open(filename, "w") as f:
            for line in self.level:
                f.write(line + "\n")
        return filename

    @classmethod
    def load(cls, filename: str) -> "SampleOutput":
        level = load_level(filename)
        return SampleOutput(level=level)

    def play(self):
        from mario_gpt.simulator import Simulator
        Simulator(level=self.level).interactive()

    def run_astar(self, render=True):
        from mario_gpt.simulator import Simulator
        Simulator(level=self.level).astar(render)


def _patch_positions(H: int, width: int, x_offset: int = 0):
    """
    Build (xs, ys) for `width` columns starting at x_offset, column-major.
    Returns flat lists of length H*width.
    """
    xs: List[float] = []
    ys: List[int] = []
    for c_local in range(width):
        xs += [float(x_offset + c_local)] * H
        ys += list(range(H))
    return xs, ys


class GPTSampler:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        temperature: float = 2.0,
        top_k: int = 16,
        use_tqdm: bool = False,
        use_argmax: bool = False,
        # legacy kwargs ignored
        context_len: int = 700,
        positions_2d_NonExpand=None,
    ):
        self.mario_lm = mario_lm
        self.temperature = temperature
        self.top_k = top_k
        self.use_tqdm = use_tqdm
        self.use_argmax = use_argmax
        self.logits_warper = LogitsProcessorList([
            TopKLogitsWarper(top_k),
            TemperatureLogitsWarper(temperature),
        ])

    @property
    def device(self) -> torch.device:
        return self.mario_lm.device

    def _encode_char(self, ch: str) -> int:
        return self.mario_lm.tokenizer.encode(ch, add_special_tokens=False)[0]

    def _next_token(
        self,
        input_ids: torch.Tensor,
        position_2d: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> int:
        with torch.no_grad():
            attn = torch.ones_like(input_ids)
            out = self.mario_lm.lm(
                input_ids=input_ids,
                attention_mask=attn,
                encoder_hidden_states=encoder_hidden_states,
                position_2d=position_2d,
            )
            logits = out.logits[:, -1, :]
            if self.use_argmax:
                return logits.argmax(-1).item()
            scores = self.logits_warper(input_ids, logits)
            probs = torch.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

    def _sample_patch(
        self,
        prefix_ids: List[int],
        prefix_xs: List[float],
        prefix_ys: List[int],
        struct_xs: List[float],
        struct_ys: List[int],
        encoder_hidden_states: torch.Tensor,
        H: int,
        patch_width: int,
        forced_ids: Optional[List] = None,
    ) -> List[int]:
        """
        Autoregressively generate structure tokens (H*patch_width) given prefix.

        For normal blocks:
          prefix_ids = target_path(350) + future_path(140) = 490 tokens
        For the last partial block:
          prefix_ids = target_path(350) + future_path(140) + overlap_context(n*H)
        struct_xs/ys : positions for new tokens (x=1..24 matching training format)
        forced_ids   : per-step override; None means use model output, int means force that token ID.
        """
        generated: List[int] = []
        total = H * patch_width
        for step in range(total):
            cur_ids = prefix_ids + generated
            input_ids = torch.tensor([cur_ids], dtype=torch.long, device=self.device)

            xs = prefix_xs + struct_xs[:step]
            ys = prefix_ys + struct_ys[:step]
            pos = torch.tensor(list(zip(xs, ys)), dtype=torch.float32, device=self.device).unsqueeze(0)

            tok = self._next_token(input_ids, pos, encoder_hidden_states)
            if forced_ids is not None and forced_ids[step] is not None:
                tok = forced_ids[step]
            generated.append(tok)

        return generated  # H*patch_width token IDs

    def sample(
        self,
        path_grid: Optional[List[List[str]]] = None,
        path_file: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        level_height: int = 14,
        temperature: float = None,
        top_k: int = None,
        use_tqdm: bool = None,
        # legacy kwargs ignored
        num_cols: int = None,
        seed=None,
        num_steps: int = None,
        encoder_hidden_states: torch.Tensor = None,
        return_tensor: bool = False,
        **kwargs,
    ) -> SampleOutput:
        if temperature is not None:
            self.temperature = temperature
            self.logits_warper = LogitsProcessorList([
                TopKLogitsWarper(self.top_k),
                TemperatureLogitsWarper(self.temperature),
            ])
        if top_k is not None:
            self.top_k = top_k
        if use_tqdm is not None:
            self.use_tqdm = use_tqdm

        H = level_height

        if path_grid is None:
            if path_file is not None:
                path_grid = load_path_grid(path_file, H)
            else:
                W = num_cols or PATCH_WIDTH
                path_grid = [['-'] * W for _ in range(H)]

        W = len(path_grid[0])
        assert W > FUTURE_WIDTH, (
            f"path_grid width {W} too small for FUTURE_WIDTH={FUTURE_WIDTH}"
        )

        self.mario_lm.eval()
        with torch.no_grad():
            if encoder_hidden_states is None:
                if prompts is not None:
                    enc = self.mario_lm.prompter.output_hidden(prompts[0])
                else:
                    enc = self.mario_lm.prompter(sample_prompt=True)[1]
                encoder_hidden_states = enc.view(1, 1, -1).to(self.device)

        # base position templates (x=0~24 for target path, x=25~34 for future path)
        target_xs_base, target_ys_base = _patch_positions(H, PATCH_WIDTH, x_offset=0)
        future_xs_base, future_ys_base = _patch_positions(H, FUTURE_WIDTH, x_offset=PATCH_WIDTH)

        # tile_columns[g] = generated token list for global col g.
        tile_columns: List[List[int]] = []

        # Block 0: full 25-col AR over x=0..24, no struct context.
        # Block k>=1: 1-col context (last generated col at x=0) + 24 new cols (x=1..24).
        block_starts: List[tuple] = []
        if PATCH_WIDTH + FUTURE_WIDTH <= W:
            block_starts.append((0, 0))  # block 0: new_start=0, path_start=0
        k = 1
        while True:
            new_start  = PATCH_WIDTH + (k - 1) * BLOCK_NEW   # 25, 49, 73, ...
            path_start = new_start - BLOCK_CONTEXT             # 24, 48, 72, ...
            if path_start + PATCH_WIDTH + FUTURE_WIDTH > W:
                # last partial block: shift path_start to fit
                path_start_last = W - PATCH_WIDTH - FUTURE_WIDTH
                if path_start_last >= 0 and new_start < W - FUTURE_WIDTH:
                    block_starts.append((new_start, path_start_last))
                break
            if new_start >= W - FUTURE_WIDTH:
                break
            block_starts.append((new_start, path_start))
            k += 1

        block_iter = tqdm(block_starts, desc="Generating blocks") if self.use_tqdm else block_starts

        with torch.no_grad():
            for new_start, path_start in block_iter:
                # target pure path: 25 cols from path_start, local x=0~24
                target_ids: List[int] = []
                for c_local in range(PATCH_WIDTH):
                    c_global = path_start + c_local
                    for r in range(H):
                        target_ids.append(self._encode_char(path_grid[r][c_global]))

                # future pure path: 10 cols past target, local x=25~34
                future_ids: List[int] = []
                for c_local in range(FUTURE_WIDTH):
                    c_global = path_start + PATCH_WIDTH + c_local
                    for r in range(H):
                        future_ids.append(self._encode_char(path_grid[r][c_global]))

                # struct context: cols in [path_start, new_start) already generated
                # block 0: range is empty (no context)
                # block k>=1: range = [path_start] = 1 col at x=0
                context_ids: List[int] = []
                context_xs: List[float] = []
                context_ys: List[int] = []
                for g_col in range(path_start, new_start):
                    if g_col < len(tile_columns):
                        x = float(g_col - path_start)
                        context_ids.extend(tile_columns[g_col])
                        context_xs += [x] * H
                        context_ys += list(range(H))

                prefix_ids = target_ids + future_ids + context_ids
                prefix_xs  = target_xs_base + future_xs_base + context_xs
                prefix_ys  = target_ys_base + future_ys_base + context_ys

                # block 0: generate PATCH_WIDTH cols (x=0..24)
                # block k>=1: generate BLOCK_NEW cols (x=1..24), or fewer for last block
                if new_start == 0:
                    actual_new = min(PATCH_WIDTH, W - FUTURE_WIDTH)
                else:
                    actual_new = min(BLOCK_NEW, W - FUTURE_WIDTH - new_start)

                struct_xs: List[float] = []
                struct_ys: List[int] = []
                for i in range(actual_new):
                    x = float((new_start + i) - path_start)
                    struct_xs += [x] * H
                    struct_ys += list(range(H))

                # precompute per-step forced token IDs from path_grid
                forced_ids = [None] * (H * actual_new)
                if path_grid is not None:
                    for i in range(actual_new):
                        g_col = new_start + i
                        if g_col < W:
                            for r in range(H):
                                ch = path_grid[r][g_col]
                                if ch in PATH_TOKENS:
                                    forced_ids[i * H + r] = self._encode_char(ch)

                # generate block autoregressively with per-token force injection
                new_block_ids = self._sample_patch(
                    prefix_ids, prefix_xs, prefix_ys,
                    struct_xs, struct_ys,
                    encoder_hidden_states, H, patch_width=actual_new,
                    forced_ids=forced_ids,
                )

                for i in range(actual_new):
                    tile_columns.append(list(new_block_ids[i * H: (i + 1) * H]))

        self.mario_lm.train()

        W_used = len(tile_columns)

        # reconstruct 2D level rows
        level_rows: List[str] = []
        for r in range(H):
            row = "".join(
                self.mario_lm.tokenizer.decode([tile_columns[c][r]])
                for c in range(W_used)
            )
            level_rows.append(row)

        return SampleOutput.from_rows(level_rows, self.mario_lm.tokenizer,
                                      self.mario_lm.prompter)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class BertSampler:
    pass
