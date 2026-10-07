"""
LevelGenEnv —— sampled-action MDP 環境（對齊 gumbel-alphazero-mariogpt-proposal S2）。

state s_t = (已生成的 tile_columns, 條件 c)
action a  = 一個 SBS segment（seg_cols 欄），由 propose_segment 提出
transition: st+1 = st ⊕ a（append segment，確定性）
terminal  : tile_columns 達 target_cols
reward    : r_T = 1[A* solvable] + λs·AdhSTRUCT + λp·AdhPATH（terminal-only）

設計要點：
  - step 前進 seg_cols 欄，重建 block-sliding 的 prefix（target/future path + struct context）
  - A* 可切換 mock / 真（阶段一用 mock 跑通邏輯，5 秒的真 A* 只在驗證數值時開）
  - 復用 sampler 的 _encode_char / _patch_positions / propose_segment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# 對齊 sampler.py / dataset.py
from mario_gpt_structure.sampler import PATCH_WIDTH, FUTURE_WIDTH, PATH_TOKENS, _patch_positions

from mario_gpt_structure.adherence import terminal_reward, PATH_TOKENS as ADH_PATH_TOKENS


# ---- A* 切換 ----
def mock_solvable(level_rows: List[str], H: int = 14) -> bool:
    """
    阶段一用的假 A*：簡單啟發式，跳過 5 秒真 A*。
    規則：最底列（地面）大致連續、無過寬斷崖 → 當可通關。
    只為驗證 pipeline 控制流，非真實可玩性。
    """
    ground = level_rows[-1]  # 最底列
    W = len(ground)
    max_gap = 0
    cur_gap = 0
    for c in range(W):
        if ground[c] in ('-',):   # 空 = 斷崖
            cur_gap += 1
            max_gap = max(max_gap, cur_gap)
        else:
            cur_gap = 0
    return max_gap <= 4  # 斷崖不超過 4 欄就當可跳過


def real_solvable(level_rows: List[str], timeout: float = 60.0) -> bool:
    """真 A*（5 秒/次）。驗證 reward 數值時用。"""
    from mario_gpt_structure.astar_simulator import run_astar_solvable
    return run_astar_solvable(level_rows, timeout=timeout)


@dataclass
class LevelState:
    """MDP 的 state：已生成的 tile_columns + 目前欄數。"""
    tile_columns: List[List[int]] = field(default_factory=list)  # [num_cols][H] token id

    @property
    def num_cols(self) -> int:
        return len(self.tile_columns)

    def clone(self) -> "LevelState":
        return LevelState(tile_columns=[col[:] for col in self.tile_columns])


class LevelGenEnv:
    def __init__(
        self,
        sampler,                       # SBSSampler（含 propose_segment）
        path_grid: List[List[str]],    # [H][W] 指定 path（PATH_TOKEN 或 '-'）
        target_prompt: str,            # STRUCT 條件，如 "many pipes, some blocks, low elevation"
        encoder_hidden_states: torch.Tensor,
        seg_cols: int = 2,             # 一個 action = 幾欄
        target_cols: Optional[int] = None,   # 生成到幾欄終端（阶段一設短，如 6-8）
        H: int = 14,
        K: int = 4,
        lambda_s: float = 0.5,
        lambda_p: float = 0.5,
        use_mock_astar: bool = True,   # 阶段一 True
    ):
        self.sampler = sampler
        self.path_grid = path_grid
        self.target_prompt = target_prompt
        self.encoder_hidden_states = encoder_hidden_states
        self.seg_cols = seg_cols
        self.H = H
        self.K = K
        self.lambda_s = lambda_s
        self.lambda_p = lambda_p
        self.use_mock_astar = use_mock_astar

        self.W = len(path_grid[0])
        # 目標欄數：預設用 path_grid 可生成的範圍（留 FUTURE_WIDTH 尾巴）
        self.target_cols = target_cols or (self.W - FUTURE_WIDTH)

        # 位置模板（對齊 sampler）
        self._target_xs, self._target_ys = _patch_positions(H, PATCH_WIDTH, x_offset=0)
        self._future_xs, self._future_ys = _patch_positions(H, FUTURE_WIDTH, x_offset=PATCH_WIDTH)

    # ---------------- MDP 介面 ----------------
    def reset(self) -> LevelState:
        return LevelState(tile_columns=[])

    def is_terminal(self, state: LevelState) -> bool:
        return state.num_cols >= self.target_cols

    def propose_actions(self, state: LevelState):
        """
        在當前 state 呼叫 SBS，回傳 K 個 segment 候選（動作集合 A(s_t)）。
        每個候選是 seg_cols 欄的 structural segment。
        """
        prefix_ids, prefix_xs, prefix_ys, struct_xs, struct_ys, forced_ids = \
            self._build_prefix(state)

        candidates = self.sampler.propose_segment(
            prefix_ids=prefix_ids,
            prefix_xs=prefix_xs,
            prefix_ys=prefix_ys,
            struct_xs=struct_xs,
            struct_ys=struct_ys,
            encoder_hidden_states=self.encoder_hidden_states,
            forced_ids=forced_ids,
            H=self.H,
            seg_cols=self._cur_seg_cols(state),
            K=self.K,
        )
        return candidates

    def step(self, state: LevelState, candidate) -> LevelState:
        """
        把選中的 segment candidate append 進 state（確定性 transition）。
        candidate.tile_columns : [seg_cols][H]
        """
        new_state = state.clone()
        for col in candidate.tile_columns:
            new_state.tile_columns.append(list(col))
        # 截到 target_cols（最後一段可能超出）
        if new_state.num_cols > self.target_cols:
            new_state.tile_columns = new_state.tile_columns[: self.target_cols]
        return new_state

    def terminal_reward(self, state: LevelState) -> Dict[str, float]:
        """終端 reward：組 level_rows → A*（mock/真）+ adherence。"""
        level_rows = self._to_level_rows(state)
        solvable = (mock_solvable(level_rows, self.H) if self.use_mock_astar
                    else real_solvable(level_rows))
        return terminal_reward(
            level_rows=level_rows,
            target_prompt=self.target_prompt,
            path_grid=self.path_grid,
            prompter=self.sampler.mario_lm.prompter,
            astar_solvable=solvable,
            lambda_s=self.lambda_s,
            lambda_p=self.lambda_p,
            H=self.H,
        )

    # ---------------- 內部：重建 block-sliding prefix ----------------
    def _cur_seg_cols(self, state: LevelState) -> int:
        """這一段實際要生成幾欄（最後一段可能不足 seg_cols）。"""
        remaining = self.target_cols - state.num_cols
        return max(1, min(self.seg_cols, remaining))

    def _build_prefix(self, state: LevelState):
        """
        依當前已生成欄數，重建下一段的 prefix：
          target_path(25欄) + future_path(10欄) + struct_context(視窗內已生成的欄)
        對齊 sampler.sample / 訓練格式：視窗內的 structure 以 x=0..24 自回歸生成，
        要生成的欄前面所有已生成欄都必須放進 context（否則後段看不到前段 structure）。

        視窗起點 path_start = max(0, new_start + seg - PATCH_WIDTH)：
          - new_start + seg <= 25：path_start = 0（同 sampler block 0），context = 已生成的 [0, new_start)
          - 超過 25 欄：視窗右移，讓這一段最後一欄落在 x=24（同 sampler block k>=1），
            context = 視窗內已生成的 [path_start, new_start)，x 不會超出訓練範圍 0..24
        """
        H = self.H
        new_start = state.num_cols               # 下一段從這欄開始
        seg = self._cur_seg_cols(state)
        assert seg <= PATCH_WIDTH, f"seg_cols={seg} > PATCH_WIDTH={PATCH_WIDTH}"

        path_start = max(0, new_start + seg - PATCH_WIDTH)

        # target pure path: PATCH_WIDTH 欄，從 path_start 讀 path_grid
        target_ids: List[int] = []
        for c_local in range(PATCH_WIDTH):
            c_global = path_start + c_local
            for r in range(H):
                ch = self.path_grid[r][c_global] if c_global < self.W else '-'
                path_ch = ch if ch in PATH_TOKENS else '-'
                target_ids.append(self.sampler._encode_char(path_ch))

        # future pure path: FUTURE_WIDTH 欄
        future_ids: List[int] = []
        for c_local in range(FUTURE_WIDTH):
            c_global = path_start + PATCH_WIDTH + c_local
            for r in range(H):
                ch = self.path_grid[r][c_global] if c_global < self.W else '-'
                path_ch = ch if ch in PATH_TOKENS else '-'
                future_ids.append(self.sampler._encode_char(path_ch))

        # struct context：視窗內已生成的欄 [path_start, new_start)，x = 欄在視窗中的位置
        context_ids: List[int] = []
        context_xs: List[float] = []
        context_ys: List[int] = []
        for g_col in range(path_start, new_start):
            context_ids.extend(state.tile_columns[g_col])
            context_xs += [float(g_col - path_start)] * H
            context_ys += list(range(H))

        prefix_ids = target_ids + future_ids + context_ids
        prefix_xs = self._target_xs + self._future_xs + context_xs
        prefix_ys = self._target_ys + self._future_ys + context_ys

        # 要生成的 seg 欄的位置（local x 從 new_start-path_start 起，最大 24）
        struct_xs: List[float] = []
        struct_ys: List[int] = []
        for i in range(seg):
            x = float((new_start + i) - path_start)
            struct_xs += [x] * H
            struct_ys += list(range(H))

        # forced_ids：path token 位置強制填
        forced_ids: List[Optional[int]] = [None] * (H * seg)
        for i in range(seg):
            g_col = new_start + i
            if g_col < self.W:
                for r in range(H):
                    ch = self.path_grid[r][g_col]
                    if ch in PATH_TOKENS:
                        forced_ids[i * H + r] = self.sampler._encode_char(ch)

        return prefix_ids, prefix_xs, prefix_ys, struct_xs, struct_ys, forced_ids

    def _to_level_rows(self, state: LevelState) -> List[str]:
        """把 tile_columns 組成 14 個 row 字串（供 A* 和 adherence）。"""
        tok = self.sampler.mario_lm.tokenizer
        H = self.H
        W = state.num_cols
        rows: List[str] = []
        for r in range(H):
            row = "".join(tok.decode([state.tile_columns[c][r]]) for c in range(W))
            assert len(row) == W, f"row {r} len {len(row)} != {W}（有 token decode 不是 1 字元）: {row!r}"
            rows.append(row)
        return rows
