import numpy as np

H = 14
W = 12  # 先用小寬度驗證；換成 7295 也可（輸出就別全列印）

def characterize(level_rows):
    # level_rows: list[str]，長度 H，每列長度 W
    return [list(s) for s in level_rows]

def flip_and_transpose(arr, flip_first=False):
    # arr: (H,W)
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr

def join_list_of_list(str_lists):
    return ["".join(s) for s in str_lists]

def forward_flatten(level_rows):
    # 模擬 MarioDataset.convert_level_to_tensor 的 2D→1D 核心
    ft = np.array(characterize(level_rows))      # (H,W)
    arr = flip_and_transpose(ft, flip_first=False)  # (W,H) 然後沿最後軸 flip => bottom-to-top
    s = "".join(join_list_of_list(arr))         # 逐 row 連接 => column-first 且 bottom-to-top
    return s

def inverse_unflatten(seq, H=14):
    # 給 1D 序列（長 T=H*W），還原回原始 14×W 的方向（與 forward_flatten 前的 level_rows 一致）
    T = len(seq)
    W = T // H
    out = [[""]*W for _ in range(H)]
    for t, ch in enumerate(seq):
        col = t // H
        row = (H - 1) - (t % H)  # bottom-to-top
        out[row][col] = ch
    return ["".join(row) for row in out]

def build_positions_2d(context_len, height=14):
    # 與 flatten 規則一致的座標表：x=col，y=bottom->top
    t = np.arange(context_len)
    x = t // height
    y = (height - 1) - (t % height)
    return np.stack([x, y], axis=-1)  # (T,2)

# --------- 測試資料：用可視化字元產生 14xW 的 "原圖" ----------
# 讓每個位置顯示它的 row（a..n）或 col（0..9,A..），方便肉眼比對
def make_level_rows(H=14, W=12):
    rows = []
    for r in range(H):
        row_chars = []
        for c in range(W):
            # 這裡用 col 的 base36（0-9A-Z）展示；也可改成 r 的字母來看 bottom/top
            base36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            row_chars.append(base36[c % len(base36)])
        rows.append("".join(row_chars))
    return rows

level_rows = make_level_rows(H, W)
print("原始 14xW：")
for line in level_rows: print(line)

seq = forward_flatten(level_rows)
print("\n展平後前 40 個 token（t, x=col, y=row）：")
pos2d = build_positions_2d(len(seq), H)
for t in range(min(40, len(seq))):
    print(f"t={t:3d}  col={pos2d[t,0]:3d}  row={pos2d[t,1]:2d}  ch='{seq[t]}'")

back = inverse_unflatten(seq, H)
print("\n反向還原 14xW：")
for line in back: print(line)

# 驗證一致
assert back == level_rows, "反向還原與原始不一致，展平規則不對齊！"
print("\n✅ 驗證通過：forward_flatten + inverse_unflatten 與座標映射一致")

#============================================
#check 7295
import numpy as np
import random

H = 14  # level height

# ====== 和 MarioDataset.convert_level_to_tensor 相同的 2D→1D 展平核心 ======
def characterize(rows):
    return [list(r) for r in rows]

def flip_and_transpose(arr, flip_first=False):
    # 與你的程式一致：transpose 後沿最後一軸 flip
    if arr.shape[-1] > 1:
        if flip_first:
            return np.flip(arr, -1).transpose()
        return np.flip(arr.transpose(), -1)
    return arr

def flatten_like_dataset(level_rows):
    """
    level_rows: list[str] 長度=14，每列長度=W
    流程：characterize -> flip_and_transpose -> 逐 row 串接
    結果：column-first、row由下往上（bottom->top）、column 左->右
    """
    ft = np.array(characterize(level_rows))           # (H,W)
    arr = flip_and_transpose(ft, flip_first=False)    # (W,H) then flip last axis => bottom->top
    s = "".join("".join(row) for row in arr)          # 逐 row 串接
    return s

# ====== 前向/反向工具 ======
def global_flatten(level_str_14xW):
    rows = level_str_14xW.strip("\n").split("\n")
    assert len(rows) == H, f"expect 14 rows, got {len(rows)}"
    W = len(rows[0])
    for r in rows: 
        assert len(r) == W, "row width mismatch"
    return flatten_like_dataset(rows), rows, W

def extract_window_rows(rows_14xW, start_col, win_w=50):
    """ 從原始 14×W rows 擷取 [start_col, start_col+win_w) 的 14×50 子矩形 """
    W = len(rows_14xW[0])
    assert 0 <= start_col <= W - win_w, "window exceeds width"
    return [r[start_col:start_col+win_w] for r in rows_14xW]

def t_to_col_row(t, height=14):
    """ 與 flatten 相對應的索引映射：column-first，row=bottom->top """
    col = t // height
    row = (height - 1) - (t % height)
    return col, row

# ====== 驗證單一起點 ======
def verify_one(level_str_14xW, start_idx, win_w=50):
    flat, rows, W = global_flatten(level_str_14xW)
    assert start_idx % H == 0, "start_idx 必須是 14 的倍數（對齊欄界）"
    assert 0 <= start_idx <= len(flat) - H*win_w, "start_idx + 700 超界"
    start_col = start_idx // H

    # 1) 直接切全域 flatten 片段
    slice_global = flat[start_idx:start_idx + H*win_w]

    # 2) 從原圖擷取 14×50 區塊，依相同規則展平
    win_rows = extract_window_rows(rows, start_col, win_w=win_w)  # list[str] len=14, width=50
    slice_by_window = flatten_like_dataset(win_rows)

    ok = (slice_global == slice_by_window)
    return ok, {
        "start_idx": start_idx, 
        "start_col": start_col, 
        "win_w": win_w,
        "first_40_global": slice_global[:40],
        "first_40_window": slice_by_window[:40],
    }

# ====== 隨機多點驗證 ======
def verify_many(level_str_14xW, trials=20, win_w=50, seed=0):
    flat, rows, W = global_flatten(level_str_14xW)
    max_start_col = W - win_w
    rng = random.Random(seed)
    # 僅選擇對齊欄界的 start_idx（= col*14）
    cols = [rng.randint(0, max_start_col) for _ in range(trials)]
    starts = [c*H for c in cols]

    all_ok = True
    first_fail = None
    for s in starts:
        ok, info = verify_one(level_str_14xW, s, win_w=win_w)
        if not ok:
            all_ok = False
            first_fail = info
            break

    if all_ok:
        print(f"✅ 全部通過（trials={trials}, win_w={win_w}）")
    else:
        print("❌ 發現不一致！詳情：")
        print(first_fail)
    return all_ok

# ================= 使用範例 =================
# 假設你能 import FULL_LEVEL_STR_WITH_PATHS，否則把字串丟進來即可
from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS
level_str = FULL_LEVEL_STR_WITH_PATHS

# 手動呼叫單點驗證（例如 col=100 的窗口）
ok, info = verify_one(level_str, start_idx=100*14, win_w=50)
print(ok, info)

# 隨機多點驗證
verify_many(level_str, trials=50, win_w=50, seed=42)

