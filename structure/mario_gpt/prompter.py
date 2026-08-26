from __future__ import annotations

import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from transformers import pipeline

from mario_gpt.dataset import MarioDataset
from mario_gpt.utils import view_level

DEBUG = False

STATISTICS = {
    "pipe": np.array([0.0, 1.0, 3.0]),
    "enemy": np.array([0.0, 1.0, 3.0]),
    "block": np.array([0.0, 7.0, 20.0]),
    "coin": np.array([0.0, 3.0, 8.0]),
    "rect": np.array([0.0, 1.0, 2.0]),
}

FEATURE_EXTRACTION_MODEL = "facebook/bart-base"

"""
The Prompter class analyzes Mario levels and generates text prompts with feature embeddings for conditional generation:

Key Components:

## Uses BART model for feature extraction (line 36-41)
Counts level features: pipes (<>), blocks (X, S, ?, Q), and 
elevation (line 62-94)

##Converts counts to descriptive keywords using thresholds: 
"no/little/some/many" for pipes/blocks, "low/high" for elevation

Main Function (__call__):

##Takes a level tensor and generates prompts like "some pipes, many blocks, 
high elevation"

##Returns the text prompt, BART hidden embeddings, prompt dictionary, and 
string level

##Can also sample random prompts when sample_prompt=True
This enables conditional level generation by providing semantic descriptions of desired level characteristics.
"""

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

    @property
    def pipe_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["pipe"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords
    @property
    def enemy_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["enemy"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords
    @property
    def block_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["block"]
        keywords = ["little", "little", "some", "many"]
        return thresholds, keywords
    @property
    def rect_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["rect"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords
    @property
    def coin_thresholds(self) -> Tuple[List[int], List[str]]:
        thresholds = self.statistics["coin"]
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords
        """
        # Use same thresholds as pipes for coins
        thresholds = self.statistics.get("coin", self.statistics["pipe"])
        keywords = ["no", "little", "some", "many"]
        return thresholds, keywords
        """
    def count_pipes(self, flattened_level: str) -> int:
        return flattened_level.count("<>")
    def count_enemies(self, flattened_level: str) -> int:
        return flattened_level.count("E") + flattened_level.count("B")

    def count_blocks(self, flattened_level: str) -> int:
        return np.sum([flattened_level.count(char) for char in ["X", "S", "?", "Q"]])

    def count_coins(self, flattened_level: str) -> int:
        return flattened_level.count("o")

    def count_rects2(self, flattened_level: str) -> int:
        return flattened_level.count("T")

    """
    flattened_level is a single string representing the 
    entire Mario level with rows concatenated sequentially.
    """
    def count_rects(self, flattened_level: str) -> int:
        """Count complete rectangle structures in the level
        A rectangle is defined as a complete 'T' border with '-' interior
        """
        # Convert flattened string back to 2D grid (assuming 14 rows, width varies)
        level_length = len(flattened_level)
        if level_length % 14 != 0:
            print('LLLLLLLLLLL', f'level_length={level_length}')
            return 0  # Invalid level format
        
        width = level_length // 14
        level_2d = []
        for i in range(14):
            start = i * width
            end = start + width
            level_2d.append(list(flattened_level[start:end]))
        
        rect_count = 0
        visited = set()
        
        # Scan for rectangle structures
        for row in range(14):
            for col in range(width):
                if (row, col) not in visited and level_2d[row][col] == 'T':
                    # Check if this 'T' is the top-left corner of a rectangle
                    rect_size = self._find_rectangle_size(level_2d, row, col, 14, width)
                    if rect_size > 0:
                        rect_count += 1
                        # Mark all cells of this rectangle as visited
                        for r in range(row, row + rect_size):
                            for c in range(col, col + rect_size):
                                visited.add((r, c))
        
        # Check for orphaned 'T' characters without complete rectangle structures
        total_t_chars = flattened_level.count('T')
        """
        if total_t_chars > 0 and rect_count == 0:
            print(f"Warning: Found {total_t_chars} 'T' characters but no complete rectangle structures detected in level")
            print(f"Level visualization (14x{width}):")
            for i in range(14):
                start = i * width
                end = start + width
                row = flattened_level[start:end] if start < len(flattened_level) else '-' * width
                print(f"  Row {i:2d}: {row}")
            print("End of level visualization")
        """
        return rect_count
    
    def _find_rectangle_size(self, level_2d, start_row, start_col, height, width):
        """Find if there's a complete rectangle starting at (start_row, start_col)
        Returns the size of the rectangle if found, 0 otherwise
        """
        # Try different odd rectangle sizes (9x9 down to 3x3) - largest first
        for size in range(9, 2, -2):
            if start_row + size > height or start_col + size > width:
                continue
                
            # Check if this forms a valid rectangle
            if self._is_valid_rectangle(level_2d, start_row, start_col, size):
                return size
        
        return 0
    
    def _is_valid_rectangle(self, level_2d, start_row, start_col, size):
        """Check if there's a valid rectangle structure with diagonal/X pattern at the given position"""
        try:
            # Must be odd size for diagonal patterns
            if size % 2 == 0:
                return False
                
            # Check top and bottom borders
            for c in range(start_col, start_col + size):
                if (level_2d[start_row][c] != 'T' or 
                    level_2d[start_row + size - 1][c] != 'T'):
                    return False
            
            # Check left and right borders
            for r in range(start_row, start_row + size):
                if (level_2d[r][start_col] != 'T' or 
                    level_2d[r][start_col + size - 1] != 'T'):
                    return False
            
            # Check interior for diagonal/X patterns
            interior_cells = []
            diagonal_cells = []
            anti_diagonal_cells = []
            
            for r in range(start_row + 1, start_row + size - 1):
                for c in range(start_col + 1, start_col + size - 1):
                    interior_cells.append((r, c))
                    # Main diagonal (top-left to bottom-right)
                    if (r - start_row) == (c - start_col):
                        diagonal_cells.append((r, c))
                    # Anti-diagonal (top-right to bottom-left)  
                    if (r - start_row) + (c - start_col) == size - 1:
                        anti_diagonal_cells.append((r, c))
            
            # Count T's on diagonals
            diagonal_t_count = sum(1 for r, c in diagonal_cells if level_2d[r][c] == 'T')
            anti_diagonal_t_count = sum(1 for r, c in anti_diagonal_cells if level_2d[r][c] == 'T')
            
            # Valid patterns: main diagonal, anti-diagonal, or X (both diagonals)
            has_main_diagonal = diagonal_t_count == len(diagonal_cells)
            has_anti_diagonal = anti_diagonal_t_count == len(anti_diagonal_cells) 
            has_x_pattern = has_main_diagonal and has_anti_diagonal
            
            # Check if it matches one of our valid patterns
            if has_x_pattern or has_main_diagonal or has_anti_diagonal:
                # Verify non-diagonal interior cells are dashes
                for r, c in interior_cells:
                    if (r, c) not in diagonal_cells and (r, c) not in anti_diagonal_cells:
                        if level_2d[r][c] != '-':
                            return False
                return True
            
            return False
        except IndexError:
            return False

    def _flatten_level(self, string_level: List[str]) -> str:
        return "".join(string_level)

    def pipe_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_pipes(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.pipe_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} pipes", keyword
    def enemy_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_enemies(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.enemy_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} enemies", keyword
    def block_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_blocks(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.block_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} blocks", keyword

    def rect_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_rects(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.rect_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} rects", keyword

    def coin_prompt(self, flattened_level: str, level: str) -> str:
        count = self.count_coins(flattened_level)
        keyword = f"{count}"
        if not self.use_raw_counts:
            thresholds, keywords = self.coin_thresholds
            threshold = np.digitize(count, thresholds, right=True)
            keyword = keywords[threshold]
        return f"{keyword} coins", keyword

    """
    This method analyzes the top 6 rows of a Mario level 
    to determine elevation. 
    It checks for platform/pipe characters ("X", "<", ">")
      in the upper portion - 
      if found, returns "high elevation", 
      otherwise "low elevation". 
      The tuple return provides both descriptive text and 
      a short label.
    """
    def elevation_prompt(self, flattened_level: str, level: str):
        top_levels = level[:6]  # elevation 8 and up
        for t in top_levels:
            if "X" in t or "<" in t or ">" in t:
                return "high elevation", "high"
        return "low elevation", "low"

    def output_hidden(self, prompt: str, device: torch.device = torch.device("cpu")):
        # Reducing along the first dimension to get a 768 dimensional array
        return (
            self.feature_extraction(prompt, return_tensors="pt")[0]
            .mean(0)
            .to(device)
            .view(1, -1)
        )

    """
    The dataset_statistics() method analyzes a MarioDataset to compute statistical thresholds for feature categorization:
    Process:

    1 Iterate Dataset: Loops through all samples, extracts level tensors via 
    dataset[i]

    2 Convert to String: Uses view_level() to decode tokens back to readable level, 
    then flattens to string
    3 Count Features: Counts pipes (<>) and blocks (X, S, ?, Q) in each level sample

    4 Calculate Quantiles: Computes 33rd, 66th, and 95th percentiles using 
    stats.mstats.mquantiles()

    5 Returns: Dictionary with statistical thresholds:

    {
    "pipe": [q33, q66, q95],  # e.g., [0.0, 2.0, 5.0]
    "block": [q33, q66, q95]  # e.g., [50.0, 75.0, 176.0]
    }
    This generates data-driven thresholds to replace the hardcoded STATISTICS for mapping counts to keywords like "no/little/some/many".
    """

    """
    Game map area used to count features
    dataset[0]: tokens [0:700]
    dataset[1]: tokens [14:714] → 686 tokens overlap (positions 14-699)
    dataset[2]: tokens [28:728] → 672 tokens overlap with dataset[1]
    """
    def dataset_statistics(self, dataset: MarioDataset, test_num=-1):
        enemy_counts = []
        pipe_counts = []
        block_counts = []
        rect_counts = []
        rect_counts2 = []
        coin_counts = []

        stat_num = len(dataset) if test_num <= 0 else min(len(dataset), test_num)
        print(f"------ Dataset: len(dataset): {stat_num}")

        for i in range(stat_num):
            dataset_item = dataset[i]
            if len(dataset_item) == 3:
                level, _, _ = dataset_item  # (input_ids, attention_mask, position_2d)
            else:
                level, _ = dataset_item  # (input_ids, attention_mask)
            str_level = self._flatten_level(view_level(level, dataset.tokenizer))

            enemy_count = self.count_enemies(str_level)
            pipe_count = self.count_pipes(str_level)
            block_count = self.count_blocks(str_level)
            rect_count = self.count_rects(str_level)
            rect_count2 = self.count_rects2(str_level)
            coin_count = self.count_coins(str_level)

            """
            if rect_count2 > 0:
                print(f'Level {i}: pipes={pipe_count}, blocks={block_count}, rects={rect_count}, coins={coin_count},rects2={rect_count2}')
                print('11111111111111111111111111111111111')
                print(str_level)
                print('22222222222222222222222222222222222')
                ##print(level)
                ##print('33333333333333333333333333333333333')
            """

            pipe_counts.append(pipe_count)
            enemy_counts.append(enemy_count)
            block_counts.append(block_count)
            rect_counts.append(rect_count)
            rect_counts2.append(rect_count2)
            coin_counts.append(coin_count)


        ##print(f"pipe_counts: {pipe_counts}")

        d = {"enemy": {}, "pipe": {}, "block": {}, "rect": {}, "coin": {}}

        d["pipe"] = stats.mstats.mquantiles(pipe_counts, [0.33, 0.66, 0.95])
        d["enemy"] = stats.mstats.mquantiles(enemy_counts, [0.33, 0.66, 0.95])
        d["block"] = stats.mstats.mquantiles(block_counts, [0.33, 0.66, 0.95])
        d["rect"] = stats.mstats.mquantiles(rect_counts, [0.33, 0.66, 0.95])
        d["rect2"] = stats.mstats.mquantiles(rect_counts2, [0.33, 0.66, 0.95])
        d["coin"] = stats.mstats.mquantiles(coin_counts, [0.33, 0.66, 0.95])
        return d

    def __call__(
        self, level: torch.Tensor = None, sample_prompt: bool = False
    ) -> Union[str, torch.Tensor]:
        device: torch.device = torch.device("cpu")
        if not sample_prompt:
            if level is None:
                raise ValueError("Level must be provided if sample_prompt is not true!")
            str_level = view_level(level, self.level_tokenizer)
            flattened_level = self._flatten_level(str_level)

            pipe_prompt, _ = self.pipe_prompt(flattened_level, str_level)
            enemy_prompt, _ = self.enemy_prompt(flattened_level, str_level)
            block_prompt, _ = self.block_prompt(flattened_level, str_level)
            rect_prompt, _ = self.rect_prompt(flattened_level, str_level)
            coin_prompt, _ = self.coin_prompt(flattened_level, str_level)
            elevation_prompt, _ = self.elevation_prompt(flattened_level, str_level)
            device = level.device
        else:
            str_level = None
            pipe_prompt = random.choice(["no", "little", "some", "many"]) + " pipes"
            enemy_prompt = random.choice(["no", "little", "some", "many"]) + " enemies"
            block_prompt = (
                random.choice(["little", "little", "some", "many"]) + " blocks"
            )  # levels always have blocks
            rect_prompt = random.choice(["no", "little", "some", "many"]) + " rects"
            coin_prompt, _ = random.choice(["no", "little", "some", "many"]) + " coins"
            elevation_prompt = (
                random.choice(["low", "high"]) + " elevation"
            )  # levels always have blocks

        prompt_dict = {
            "pipe": pipe_prompt,
            "enemy": enemy_prompt,
            "block": block_prompt,
            "rect": rect_prompt,
            "coin": coin_prompt,
            "elevation": elevation_prompt,
        }
        if DEBUG:
            print("Trained Promprts: PIPE, BLOCK, RECT")
            print(f"??? prompt_dict = elevation_prompt: elevation_prompt,")
        prompt = f"{pipe_prompt}, {enemy_prompt}, {block_prompt}, {rect_prompt}, {coin_prompt}, {elevation_prompt}, "
        hidden = self.output_hidden(prompt, device=device)
        return prompt, hidden, prompt_dict, str_level
