"""
改造版 Simulator：astar 回傳「是否通關」的 bool，供 LevelGenEnv 的 reward 使用。

相對原版三個修正：
  1. 讀取 subprocess stdout（原版 `_ =` 丟掉了結果）
  2. 預設 norender（搜索時跑數千次，不能開圖形界面）
  3. 加 timeout（防某些 level 讓 agent 卡住）

解析 Mario-AI-Framework 的 Game Status 與 Percentage Completion。
is_solvable() 提供硬判斷；completion() 提供可用於消融的 soft reward。
"""

import os
import re
import subprocess
import tempfile
from typing import List, Optional

from mario_gpt_structure.utils import save_level

pt = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(pt, "img/")
ASTAR_JAR_PATH = os.path.join(pt, "PlayAstar.jar")

WIN_MARKERS = ("Game Status: WIN",)
LOSE_MARKERS = ("Game Status: LOSE", "Game Status: TIME_OUT")
COMPLETION_PATTERN = re.compile(
    r"Percentage Completion:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=\s|$)"
)


def _parse_completion(out: str) -> Optional[float]:
    match = COMPLETION_PATTERN.search(out)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


class AstarSimulator:
    def __init__(self, level: List[str], astar_jar_path: Optional[str] = None):
        self.level = level
        self.astar_jar_path = astar_jar_path or ASTAR_JAR_PATH

    def _run(self, timeout: float = 60.0) -> str:
        """跑 A* agent，回傳 stdout 字串（norender）。"""
        t = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
        try:
            save_level(self.level, t.name)
            t.close()
            proc = subprocess.run(
                ["java", "-jar", self.astar_jar_path, t.name, "norender", IMAGE_PATH],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            return proc.stdout.decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired:
            return "Game Status: TIME_OUT"
        finally:
            if os.path.exists(t.name):
                os.unlink(t.name)

    def is_solvable(self, timeout: float = 60.0) -> bool:
        """回傳這個 level 是否可通關。"""
        out = self._run(timeout=timeout)
        # 明確狀態優先，缺少狀態時才以完成度備援。
        if any(m in out for m in WIN_MARKERS):
            return True
        if any(m in out for m in LOSE_MARKERS):
            return False
        completion = _parse_completion(out)
        if completion is not None:
            return completion >= 1.0
        print(f"[AstarSimulator] Unrecognized stdout:\n{out[:500]}")
        return False

    def completion(self, timeout: float = 60.0) -> float:
        """回傳完成度 ∈ [0,1]，可當 soft solvability reward。"""
        out = self._run(timeout=timeout)
        completion = _parse_completion(out)
        if completion is not None:
            return completion
        if any(m in out for m in WIN_MARKERS):
            return 1.0
        return 0.0


def run_astar_solvable(level_rows: List[str], timeout: float = 60.0) -> bool:
    """便捷函式：給 14 row level，回傳是否可通關。供 terminal_reward 使用。"""
    return AstarSimulator(level_rows).is_solvable(timeout=timeout)
