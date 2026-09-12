import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image

pt = os.path.dirname(os.path.realpath(__file__))
TILE_DIR = os.path.join(pt, "data", "tiles")


def trim_level(level):
    mod = level.shape[-1] % 14
    if mod > 0:
        return level[:, :-mod]
    return level


def characterize(str_lists):
    return [list(s[::-1]) for s in str_lists]


def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]


def view_level(level_tokens, tokenizer, flatten=False):
    ##print('mario-gpt-OK-11111111111')
    if flatten:
        return tokenizer.batch_decode(level_tokens.detach().cpu().squeeze())
    str_list = tokenizer.decode(level_tokens.detach().cpu()).replace("<mask>", "Y")
    str_list = [str_list[i : i + 14] for i in range(0, len(str_list), 14)]
    for i in range(len(str_list)):
        length = len(str_list[i])
        diff = 14 - length
        if diff > 0:
            str_list[i] = str_list[i] + "Y" * diff
    return join_list_of_list(np.array(characterize(str_list)).T)


def is_flying_enemy(array, row, col):
    num_rows = array.shape[0]
    if row == num_rows - 1:
        return False
    below = array[row + 1][col]
    return below == "-"


def char_array_to_image(array, chars2pngs, target_size=None):
    """
    Convert a 16-by-16 array of integers into a PIL.Image object
    param: array: a 16-by-16 array of integers
    """
    if target_size is None:
        image = Image.new("RGB", (array.shape[1] * 16, array.shape[0] * 16))
    else:
        image = Image.new("RGB", (target_size[1] * 16, target_size[0] * 16))
    for row in range(array.shape[0]):
        for col, char in enumerate(array[row]):
            value = chars2pngs["-"]
            if char in chars2pngs:
                value = chars2pngs[char]
            ##else:
                ##print(f"REPLACING {value}", (col, row))
            image.paste(value, (col * 16, row * 16))
    return image


def convert_level_to_png(
    level: Union[str, torch.Tensor],
    tokenizer=None,
    tiles_dir: str = None,
    target_size=None,
):
    if isinstance(level, torch.Tensor):
        level = view_level(level, tokenizer)
    if tiles_dir is None:
        tiles_dir = TILE_DIR
        
    chars2pngs = {
        "-": Image.open(f"{tiles_dir}/smb-background.png"),
        "X": Image.open(f"{tiles_dir}/smb-unpassable.png"),
        "S": Image.open(f"{tiles_dir}/smb-breakable.png"),
        "?": Image.open(f"{tiles_dir}/smb-question.png"),
        "Q": Image.open(f"{tiles_dir}/smb-question.png"),
        "o": Image.open(f"{tiles_dir}/smb-coin.png"),
        "E": Image.open(f"{tiles_dir}/smb-enemy.png"),
        "<": Image.open(f"{tiles_dir}/smb-tube-top-left.png"),
        ">": Image.open(f"{tiles_dir}/smb-tube-top-right.png"),
        "[": Image.open(f"{tiles_dir}/smb-tube-lower-left.png"),
        "]": Image.open(f"{tiles_dir}/smb-tube-lower-right.png"),
        "x": Image.open(f"{tiles_dir}/smb-path.png"),  # self-created
        "Y": Image.fromarray(
            np.uint8(np.zeros((16, 16)))
        ),  # black square,  # self-created
        "N": Image.open(f"{tiles_dir}/N.png"),  # self-created
        "B": Image.open(f"{tiles_dir}/cannon_top.png"),
        "b": Image.open(f"{tiles_dir}/cannon_bottom.png"),
        "F": Image.open(f"{tiles_dir}/flying_koopa.png"),
        "@": Image.open(f"{tiles_dir}/smb-coin.png"),
        "T": Image.open(f"{tiles_dir}/N.png"),
        
        # =========================================================
        # ⭐️ [修改]：將不同的路徑 Token 對應到各自專屬的圖片
        # =========================================================
        "M": Image.open(f"{tiles_dir}/smb-movement.png"),
        "J": Image.open(f"{tiles_dir}/smb-bigJump.png"),
        "j": Image.open(f"{tiles_dir}/smb-smalljump.png"),
        "A": Image.open(f"{tiles_dir}/smb-JumpandMove.png"),
        "C": Image.open(f"{tiles_dir}/smb-getCoin.png"),
        "w": Image.open(f"{tiles_dir}/w-j-end-2.png"),
        "W": Image.open(f"{tiles_dir}/W-J-END.png"),
        "k": Image.open(f"{tiles_dir}/k-j-in-air.png"),
        "K": Image.open(f"{tiles_dir}/K-jump in air.png"),
        "L": Image.open(f"{tiles_dir}/L.png"),
        "r": Image.open(f"{tiles_dir}/r.png"),
        "R": Image.open(f"{tiles_dir}/R-BIG.png"),
        "U": Image.open(f"{tiles_dir}/U.png"),
    }
    
    levels = [list(s) for s in level]
    arr = np.array(levels)
    return char_array_to_image(arr, chars2pngs, target_size), arr, level


def generate_timelapse(level_tensor, mario_lm, interval: int = 1):
    images = []
    full_size = math.ceil(level_tensor.shape[-1] / 14)
    for i in range(1, level_tensor.shape[-1], interval):
        img = convert_level_to_png(
            level_tensor[:i], mario_lm.tokenizer, target_size=(14, full_size)
        )[0]
        images.append(img)
    return images


def save_level(level: List[str], filename: str):
    concatenated = "\n".join(level)
    with open(filename, "w") as f:
        f.write(concatenated)
    return filename


def load_level(filename: str) -> List[str]:
    with open(filename, "r") as file:
        level_string = file.read()
    lines = level_string.split("\n")
    lines = [line.strip() for line in lines]
    return lines

def chars_to_2d_level(flat_chars: str, height: int = 14) -> str:
    """
    Convert a flat character string of length (height × width)
    into a 2D level string with each row on a new line.
    
    Args:
        flat_chars (str): A flat string (e.g., 700 characters).
        height (int): Number of rows (default = 14).
    
    Returns:
        str: Multiline string representing the 2D level.
    """
    total_len = len(flat_chars)
    assert total_len % height == 0, "Input length must be divisible by height."
    width = total_len // height
    level_rows = [
        flat_chars[r * width:(r + 1) * width]
        for r in range(height)
    ]
    return "\n".join(level_rows)

def chars_to_2d_level_column_first_bottom_to_top(flat_chars: str, height: int = 14) -> str:
    """
    Convert a flat string arranged in column-major, bottom-to-top order
    into a row-major 2D game level string.
    
    Args:
        flat_chars (str): A string of game characters (e.g., 700 chars for 14x50).
        height (int): Number of rows (default 14).
    
    Returns:
        str: Multiline string representing the level in row-major top-down format.
    """
    total_len = len(flat_chars)
    assert total_len % height == 0, "Length must be divisible by height."
    width = total_len // height

    # Create 2D array filled column-wise, bottom-to-top
    grid = [['' for _ in range(width)] for _ in range(height)]
    idx = 0
    for col in range(width):
        for row in reversed(range(height)):  # bottom to top
            grid[row][col] = flat_chars[idx]
            idx += 1

    # Join rows into string
    return "\n".join("".join(row) for row in grid)


def view_level_rowmajor(level_tokens, tokenizer, height=14):
    """
    Decode a row-major token sequence into a list of `height` row strings.
    Token t → row = t // width, col = t % width.
    """
    flat_str = tokenizer.decode(level_tokens.detach().cpu()).replace("<mask>", "Y")
    total = len(flat_str)
    if total == 0 or height == 0:
        return []
    width = total // height
    if width == 0:
        return [flat_str]
    return [flat_str[r * width : (r + 1) * width] for r in range(height)]


TOKENS = [
    "-",
    "X",
    "S",
    "?",
    "Q",
    "o",
    "E",
    "<",
    ">",
    "[",
    "]",
    "x",
    "Y",
    "N",
    "B",
    "b",
    "T",
    # ⭐️ 同步在全局 Token 列表中加入路徑符號
    "M",
    "J",
    "j",
    "A",
    "C",
]