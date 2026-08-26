from __future__ import annotations

import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

DEBUG = False #True
DEFAULT_MODEL = "distilgpt2"

"""
grep's -o will only output the matches, ignoring lines; wc can count them:

grep -o 'needle' file | wc -l
This will also match 'needles' or 'multineedle'.

To match only single words use one of the following commands:

grep -ow 'needle' file | wc -l
grep -o '\bneedle\b' file | wc -l
grep -o '\<needle\>' file | wc -l
"""

def split_given_size(a, size):
    return np.split(a, np.arange(size, len(a), size))


def flip_and_transpose(arr: np.array, flip_first: bool = False):
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def characterize(str_lists):
    return [list(s) for s in str_lists]

"""
The MarioDataset converts 2D char levels to 1D training data through the 
convert_level_to_tensor method:

1 Characterize: Converts each level string row into a list of characters 
using characterize(level)

2 Flip & Transpose: Uses flip_and_transpose(ft) to reorient the 2D array 
(flips columns then transposes to read column-wise)

3 Flatten: Joins all sublists into one continuous string with 
"".join(join_list_of_list(str_arr))

4 Tokenize: Converts the 1D string to tokens using the tokenizer

This effectively reads the 2D level column-by-column (??from right-to-left), 
creating a sequential 1D representation for GPT training.

5. see check_2d_to_1d_level.py
"""

"""
dataset[0]: tokens [0:700]
dataset[1]: tokens [14:714] → 686 tokens overlap (positions 14-699)
dataset[2]: tokens [28:728] → 672 tokens overlap with dataset[1]
"""

def build_positions_2d(context_len: int, height: int = 14, device="cpu"):
    t = torch.arange(context_len, device=device)
    x = t // height
    y = (height - 1) - (t % height)  # bottom->top
    return torch.stack([x, y], dim=-1).to(torch.float32)  # (T,2)

def build_positions_2d_1(context_len: int, height: int = 14, device="cuda"):
    width = context_len // height
    x = torch.arange(context_len, device=device) // height  # 水平
    y = torch.arange(context_len, device=device) % height   # 垂直
    return torch.stack([x, y], dim=-1)  # (T, 2)

class MarioDataset(Dataset):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        level_string: Optional[str] = None,
        context_len: int = 700,
        height: int = 14,
        remove_start_end_tokens: bool = False,
        sample_all_indices: bool = False, #True
    ):
        if level_string is None:
            print(
                "--------- No level string specified, using default string FULL_LEVEL_STR_WITH_PATHS..."
            )
            level_string = FULL_LEVEL_STR_WITH_PATHS
        elif ".txt" in level_string:
            print(f"-------- Loading level string from file...{level_string}")
            with open(level_string, "r") as file:
                level_string = file.read()

        self.character_set = set(level_string)

        print(f"==================================")
        print(f"Character set: {self.character_set}")

        if "\n" in self.character_set:
            self.character_set.remove("\n")
        self.vocab_size = len(self.character_set)
        self.sample_all_indices = sample_all_indices

        def get_training_corpus():
            yield list(level_string)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

        self.tokenizer = tokenizer
        if getattr(tokenizer, "train_new_from_iterator", None) is not None:
            print('-----------tokenizer: train_new_from_iterator')
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), 52000
            )
        elif getattr(tokenizer, "train_from_iterator", None) is not None:
            print('-----------tokenizer: train_from_iterator')
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
            self.tokenizer = self.tokenizer.train_new_from_iterator(
                get_training_corpus(), self.vocab_size
            )
        self.context_len = context_len
        self.height = height

        x, self.str_arr = self.convert_level_to_tensor(level_string.split("\n"))
        self.input_ids = x["input_ids"].squeeze()
        self.attention_masks = x["attention_mask"].squeeze()
        if remove_start_end_tokens:
            self.input_ids = self.input_ids[1:-1]
            self.attention_masks = self.attention_masks[1:-1]

        #======================================================
        # Generate 2D position encoding for the entire sequence
        #total_tokens = self.input_ids.shape[0]
        #width = self.height * (self.context_len // self.height)  # usually 28
        #
        #x = torch.arange(total_tokens) % width
        #y = torch.arange(total_tokens) // width
        #self.positions_2d_full = torch.stack([x, y], dim=-1)  # (N, 2)
        self.positions_2d_full = build_positions_2d(self.context_len, self.height)
        
        if DEBUG:
            print(f'dataset:self.positions_2d_full:{self.positions_2d_full.shape}')
        #=======================================================
        #
        self.indices = self.generate_indices()

        self.unique_tokens, self.unique_counts = self.input_ids.unique(
            return_counts=True
        )
        self.weighted_unique_counts = (
            1.0 / self.unique_counts / torch.sum(self.unique_counts)
        )

        self.token_dict = {}
        string_tokens = list(self.tokenizer.decode(self.unique_tokens))
        for int_token, string_token in zip(self.unique_tokens, string_tokens):
            self.token_dict[string_token] = int_token

    def convert_level_to_tensor(self, level: List[str]):
        
        ##with open("a3.txt", "w") as file:
        # Log the level input
        #print(f'5555555555 level {level}')  # This will print to console
        #file.write(f'5555555555 level {level}\n')  # Write to file

        # Characterize the level
        ft = np.array(characterize(level))
        #print(f'6666666666 characterize {ft}')  # This will print to console
        #file.write(f'6666666666 characterize {ft}\n')  # Write to file

        # Flip and transpose the characterized array
        str_arr = flip_and_transpose(ft)
        #print('77777777777 flip_transpose ', str_arr)  # This will print to console
        #file.write(f'77777777777 flip_transpose {str_arr}\n')  # Write to file

        # Join the list of lists into a single string
        str_arr = "".join(join_list_of_list(str_arr))
        #print('8888888888 join_list ', str_arr)  # This will print to console
        #file.write(f'8888888888 join_list {str_arr}\n')  # Write to file

        # Tokenize the string
        x = self.tokenizer(str_arr, return_tensors="pt")
        #print('9999999999 tokenizer  ', x)  # This will print to console
        #file.write(f'9999999999 tokenizer {x}\n')  # Write to file

        return x, str_arr



    def __len__(self):
        return self.indices.shape[0]

    def __getitem_0__(self, idx):
        if isinstance(idx, int):
            indices = self.indices[idx]
        else:
            indices = torch.stack([self.indices[i] for i in idx])
        return self.input_ids[indices], self.attention_masks[indices]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            indices = self.indices[idx]                     # (T,)
            pos2d = self.positions_2d_full                  # (T,2) 視窗內相對座標
        else:
            indices = torch.stack([self.indices[i] for i in idx])  # (B,T)
            # 為每個 batch 複製一份 (T,2) -> (B,T,2)
            pos2d = self.positions_2d_full.unsqueeze(0).repeat(indices.shape[0], 1, 1)

        return (
            self.input_ids[indices],        # 依舊用全域 indices 取 token
            self.attention_masks[indices],  # 同上
            pos2d,                          # 由於每個14x50都相同，這參數不用了。#視窗相對座標（不再用 indices 去切）
        )

    def generate_indices(self):
        out = []
        if DEBUG:
            print(f"Dataset sample_all_indices={self.sample_all_indices}")
        for idx in range(self.input_ids.shape[0] - self.context_len):
            if idx % self.height == 0 or self.sample_all_indices:
                arange = torch.arange(idx, idx + self.context_len)
                out.append(arange)
        return torch.stack(out)

    def generate_indices1(self, token_ratio=0.3):
        out = []
        if DEBUG:
            print(f"Dataset sample_all_indices={self.sample_all_indices},token_ratio={token_ratio}")
        for idx in range(self.input_ids.shape[0] - self.context_len):
            # 70% column 對齊 + 30% 隨機位置
            if idx % self.height == 0 or (self.sample_all_indices and np.random.random() < token_ratio):
                arange = torch.arange(idx, idx + self.context_len)
                out.append(arange)
        return torch.stack(out)

    def sample_indices(self, batch_size):
        out = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, self.__len__() - self.context_len)
            indices = torch.arange(start_idx, start_idx + self.context_len)
            out.append(indices)
        return torch.stack(out)

    def __str__(self):
        str_list = characterize(self.tokenizer.batch_decode(self.x["input_ids"]))
        string = "\n".join(
            join_list_of_list(flip_and_transpose(np.array(str_list), True))
        )
        return string
