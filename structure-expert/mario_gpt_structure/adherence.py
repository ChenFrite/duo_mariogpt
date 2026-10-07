"""
Adherence 計算：AdhSTRUCT + AdhPATH，供 LevelGenEnv 的終端 reward 使用。

AdhSTRUCT：復用 prompter 的 counting method + np.digitize 量化。
  量化 index 有序（0=no/little ... 3=many），直接算序數距離 → soft adherence。
AdhPATH  ：path token 由 forced 保證位置正確，adherence 主要檢查
  「非指定位置有沒有冒出多餘 path token」。

先用簡單作法（本檔）。之後若要 cascaded / multi-cross-attention 的 slot 版本
再換 prompter，本檔的 adherence 邏輯不變（仍是量化詞比對）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# 對齊 sampler.py / dataset.py
PATH_TOKENS = frozenset({
    'M', 'J', 'j', 'A', 'C', 'L', 'K', 'W', 'U', 'k', 'w', 'R', 'r',
})

# feature → (count_method_name, threshold_key)
# elevation 特殊處理（非計數）
STRUCT_FEATURES = ["pipe", "enemy", "block", "coin"]


def _quantize_index(count: int, thresholds: np.ndarray) -> int:
    """對齊 prompter 的 np.digitize(count, thresholds, right=True)，回傳有序 index。"""
    return int(np.digitize(count, thresholds, right=True))


def struct_quant_indices(level_rows: List[str], prompter) -> Dict[str, int]:
    """
    用 prompter 的 counting method 算生成 level 每個 feature 的量化 index。
    level_rows: 14 個 row 字串（view_level 格式）。
    回傳 {feature: quant_index}，index 越大代表越多。
    """
    flat = "".join(level_rows)
    counts = {
        "pipe": prompter.count_pipes(flat),
        "enemy": prompter.count_enemies(flat),
        "block": prompter.count_blocks(flat),
        "coin": prompter.count_coins(flat),
    }
    out: Dict[str, int] = {}
    for feat in STRUCT_FEATURES:
        thresholds = prompter.statistics[feat]
        out[feat] = _quantize_index(counts[feat], thresholds)

    # elevation：對齊 prompter.elevation_prompt（top 6 rows 有無 X/</>）
    top = level_rows[:6]
    out["elevation"] = 1 if any(("X" in t or "<" in t or ">" in t) for t in top) else 0
    return out


# 量化詞 → 有序 index（用於解析 target prompt）
_KW_ORDER = {"no": 0, "little": 1, "some": 2, "many": 3}
_ELEV_ORDER = {"low": 0, "high": 1}


def parse_target_indices(target_prompt: str) -> Dict[str, int]:
    """
    從 target prompt 字串解析每個 feature 的目標量化 index。
    例："many pipes, some blocks, low elevation"
        → {"pipe": 3, "block": 2, "elevation": 0}
    只回傳 prompt 有指定的 feature（其餘視為 don't-care，不計入 adherence）。
    """
    out: Dict[str, int] = {}
    for item in target_prompt.split(","):
        item = item.strip().lower()
        if not item:
            continue
        # 找量化詞
        if "elevation" in item:
            for kw, idx in _ELEV_ORDER.items():
                if kw in item:
                    out["elevation"] = idx
            continue
        for feat_kw, feat in [("pipe", "pipe"), ("enem", "enemy"),
                              ("block", "block"), ("coin", "coin")]:
            if feat_kw in item:
                for kw, idx in _KW_ORDER.items():
                    if item.startswith(kw + " ") or f" {kw} " in item or item.startswith(kw):
                        out[feat] = idx
                        break
    return out


def adh_struct(
    level_rows: List[str],
    target_prompt: str,
    prompter,
    soft: bool = True,
) -> float:
    """
    AdhSTRUCT ∈ [0,1]。
    soft=True ：量化 index 的序數距離（給搜索連續梯度，推薦）。
    soft=False：精確匹配比例（對齊論文 prompt accuracy 表）。
    只計入 target_prompt 有指定的 feature。
    """
    target = parse_target_indices(target_prompt)
    if not target:
        return 1.0  # 無 struct 指定 → 視為滿足

    gen = struct_quant_indices(level_rows, prompter)

    if not soft:
        matches = sum(1 for f, t in target.items() if gen.get(f, -1) == t)
        return matches / len(target)

    # soft：序數距離
    # 每個 feature 的最大距離：block/coin/pipe/enemy 是 3（0..3），elevation 是 1
    total_dist = 0.0
    max_dist = 0.0
    for f, t in target.items():
        g = gen.get(f, 0)
        span = 1 if f == "elevation" else 3
        total_dist += abs(g - t)
        max_dist += span
    return 1.0 - (total_dist / max_dist if max_dist > 0 else 0.0)


def adh_path(
    level_rows: List[str],
    path_grid: List[List[str]],
    H: int = 14,
) -> float:
    """
    AdhPATH ∈ [0,1]。
    path token 位置由 forced 保證正確，故 adherence 主要罰「多餘 path token」：
      非指定 path 位置（path_grid 為非 PATH_TOKEN）卻在生成 level 冒出 path token。

    level_rows: 生成 level 的 14 row 字串。
    path_grid : [H][W] 指定的 path（PATH_TOKEN 或 '-'）。
    """
    W = min(len(level_rows[0]), len(path_grid[0]))
    total = 0
    violations = 0
    for r in range(H):
        for c in range(W):
            gen_ch = level_rows[r][c]
            spec_ch = path_grid[r][c]
            spec_is_path = spec_ch in PATH_TOKENS
            gen_is_path = gen_ch in PATH_TOKENS
            if spec_is_path:
                # 指定要 path：forced 應保證，檢查有無被破壞
                total += 1
                if gen_ch != spec_ch:
                    violations += 1
            else:
                # 未指定 path：不該冒出 path token
                if gen_is_path:
                    total += 1
                    violations += 1
    if total == 0:
        return 1.0
    return 1.0 - violations / total


def terminal_reward(
    level_rows: List[str],
    target_prompt: str,
    path_grid: List[List[str]],
    prompter,
    astar_solvable: bool,
    lambda_s: float = 0.5,
    lambda_p: float = 0.5,
    H: int = 14,
) -> Dict[str, float]:
    """
    r_T = 1[A* solvable] + λs·AdhSTRUCT + λp·AdhPATH
    回傳分項與總和，方便除錯與 logging。
    """
    a_struct = adh_struct(level_rows, target_prompt, prompter, soft=True)
    a_path = adh_path(level_rows, path_grid, H)
    solvable = 1.0 if astar_solvable else 0.0
    total = solvable + lambda_s * a_struct + lambda_p * a_path
    return {
        "reward": total,
        "solvable": solvable,
        "adh_struct": a_struct,
        "adh_path": a_path,
    }
