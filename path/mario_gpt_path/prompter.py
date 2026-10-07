from __future__ import annotations

import os, sys

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import pipeline

from mario_gpt_path.dataset import MarioDataset
from mario_gpt_path.utils import view_level

DEBUG = False

# 這裡的預設數值是「佔位符」，請務必執行 dataset_statistics() 
# 來根據你真實的軌跡資料重新計算 [33%, 66%, 95%] 的分位數
STATISTICS = {
    "M": np.array([0.0, 10.0, 30.0]),  # 平移
    "J": np.array([0.0, 5.0, 15.0]),   # 大跳
    "j": np.array([0.0, 5.0, 15.0]),   # 小跳
    "A": np.array([0.0, 5.0, 15.0]),   # 空中
    "C": np.array([0.0, 2.0, 8.0]),    # 吃金幣
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"

class Prompter:
    def __init__(
        self,
        level_tokenizer,
        prompter_model: str = FEATURE_EXTRACTION_MODEL,
        use_raw_counts: bool = False,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        self.prompter_model = prompter_model
        self.feature_extraction = pipeline(
            "feature-extraction",
            model=prompter_model,
            tokenizer=prompter_model,
            framework="pt",
        )

        self.level_tokenizer = level_tokenizer

        self.use_raw_counts = use_raw_counts
        self.statistics = statistics
        if statistics is None:
            self.statistics = STATISTICS

    # ==========================================
    # Thresholds (定義 no, little, some, many)
    # ==========================================
    @property
    def M_thresholds(self) -> Tuple[List[float], List[str]]:
        return self.statistics["M"], ["no", "little", "some", "many"]

    @property
    def J_thresholds(self) -> Tuple[List[float], List[str]]:
        return self.statistics["J"], ["no", "little", "some", "many"]

    @property
    def j_thresholds(self) -> Tuple[List[float], List[str]]:
        return self.statistics["j"], ["no", "little", "some", "many"]

    @property
    def A_thresholds(self) -> Tuple[List[float], List[str]]:
        return self.statistics["A"], ["no", "little", "some", "many"]

    @property
    def C_thresholds(self) -> Tuple[List[float], List[str]]:
        return self.statistics["C"], ["no", "little", "some", "many"]

    # ==========================================
    # Count Methods (計算各個動作 Token 的數量)
    # ==========================================
    def count_M(self, flattened_level: str) -> int:
        return flattened_level.count("M")

    def count_J(self, flattened_level: str) -> int:
        return flattened_level.count("J")

    def count_j(self, flattened_level: str) -> int:
        return flattened_level.count("j")

    def count_A(self, flattened_level: str) -> int:
        return flattened_level.count("A")

    def count_C(self, flattened_level: str) -> int:
        return flattened_level.count("C")

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    # ==========================================
    # Prompt Generation (生成對應的文字 Prompt)
    # ==========================================
    def M_prompt(self, flattened_level: str) -> Tuple[str, str]:
        count = self.count_M(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.M_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} moves", keyword

    def J_prompt(self, flattened_level: str) -> Tuple[str, str]:
        count = self.count_J(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.J_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} big jumps", keyword

    def j_prompt(self, flattened_level: str) -> Tuple[str, str]:
        count = self.count_j(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.j_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} small jumps", keyword

    def A_prompt(self, flattened_level: str) -> Tuple[str, str]:
        count = self.count_A(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.A_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} air moves", keyword

    def C_prompt(self, flattened_level: str) -> Tuple[str, str]:
        count = self.count_C(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.C_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} coins collected", keyword

    # ==========================================
    # BART Embedding Extraction
    # ==========================================
    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    # ==========================================
    # Dataset Statistics (根據真實資料集動態計算 Threshold)
    # ==========================================
    def dataset_statistics(self, dataset: MarioDataset, test_num=-1):
        M_counts, J_counts, j_counts, A_counts, C_counts = [], [], [], [], []

        stat_num = len(dataset) if test_num <= 0 else min(len(dataset), test_num)
        print(f"------ Dataset: len(dataset): {stat_num}")

        for i in range(stat_num):
            dataset_item = dataset[i]
            if len(dataset_item) == 3:
                level, _, _ = dataset_item
            else:
                level, _ = dataset_item
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            M_counts.append(self.count_M(str_level))
            J_counts.append(self.count_J(str_level))
            j_counts.append(self.count_j(str_level))
            A_counts.append(self.count_A(str_level))
            C_counts.append(self.count_C(str_level))

        d = {}
        d["M"] = stats.mstats.mquantiles(M_counts, [0.33, 0.66, 0.95])
        d["J"] = stats.mstats.mquantiles(J_counts, [0.33, 0.66, 0.95])
        d["j"] = stats.mstats.mquantiles(j_counts, [0.33, 0.66, 0.95])
        d["A"] = stats.mstats.mquantiles(A_counts, [0.33, 0.66, 0.95])
        d["C"] = stats.mstats.mquantiles(C_counts, [0.33, 0.66, 0.95])
        
        return d

    # ==========================================
    # __call__ (主執行函式)
    # ==========================================
    def __call__(
        self, level: torch.Tensor = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor, Dict, List[str]]:
        device: torch.device = torch.device("cpu")
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)

            prompt_M, _ = self.M_prompt(flattened_level)
            prompt_J, _ = self.J_prompt(flattened_level)
            prompt_j, _ = self.j_prompt(flattened_level)
            prompt_A, _ = self.A_prompt(flattened_level)
            prompt_C, _ = self.C_prompt(flattened_level)
            
            device = level.device
        else:
            str_level = None
            kw = ["no", "little", "some", "many"]
            prompt_M = f"{random.choice(kw)} moves"
            prompt_J = f"{random.choice(kw)} big jumps"
            prompt_j = f"{random.choice(kw)} small jumps"
            prompt_A = f"{random.choice(kw)} air moves"
            prompt_C = f"{random.choice(kw)} coins collected"

        prompt_dict = {
            "M": prompt_M,
            "J": prompt_J,
            "j": prompt_j,
            "A": prompt_A,
            "C": prompt_C,
        }

        # 將所有條件組合成一個長句交給 BART
        prompt = f"{prompt_M}, {prompt_J}, {prompt_j}, {prompt_A}, {prompt_C}"
        
        if DEBUG:
            print(f"Generated Prompt: {prompt}")

        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level