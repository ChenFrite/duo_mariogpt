# Path Expert — 腳本說明

---

## `train.py` — 單輪訓練（快速測試用）

從頭或從 checkpoint 開始，跑一次固定步數的訓練。

- 起始模型：`BASE = "random"`（可改成 checkpoint 路徑）
- Dataset：`FULL_LEVEL_STR_WITH_PATHS`（含路徑 token 的關卡字串）
- 輸出至：`Mario-GPT2-700-context-length_29/`
- 預設：`total_steps=1001`，每 1000 步存一次

```python
# 設定區（可修改）
BASE       = "random"             # 或改為 checkpoint 路徑
OUTPUT_DIR = "Mario-GPT2-700-context-length_29"
```

```bash
python train.py
```

---

## `diff_LR_train.py` — 多輪訓練（從頭開始）

從隨機初始化開始，進行多輪連續訓練。每輪結束後自動載入上一輪的最後 checkpoint 繼續，並降低 LR。

- Round 1：`model_path = "random"`，LR = 5e-4
- Round 2+：載入上一輪最後 checkpoint，LR = 1e-4
- Dataset：`FULL_LEVEL_STR_WITH_PATHS`
- 輸出至：`M_multiround_round{N}/`

```python
# 設定區（可修改）
NUM_ROUNDS      = 10
STEPS_PER_ROUND = 10001
LR_FIRST        = 5e-4
LR_CONTINUE     = 1e-4
SAVE_ITERATION  = 10000
```

```bash
python diff_LR_train.py
```

---

## `multi_test.py` — 批次生成路徑地圖

載入訓練好的 Path Expert，批次生成多個路徑地圖並輸出報告。

- Prompt 由路徑 token 統計組成（M/J/j/A/C 的 no/little/some/many）
- 每次生成隨機選擇 prompt，生成後分析實際 token 數量是否符合
- 輸出：`multi_test/generated_level_{N}.txt`、`generated_level_{N}.png`、`prompt_check_report.txt`

```python
# 設定區（可修改）
GENERATE_COUNT = 10
CHECKPOINT_DIR = "./M_multiround_round1/iteration_10000"
```

```bash
python multi_test.py               # 預設生成 10 個
python multi_test.py --count 20    # 指定生成 20 個
```

---

## 建議使用順序

```
1. 訓練模型   →  python diff_LR_train.py
2. 生成路徑   →  python multi_test.py --count N
               （記得將 CHECKPOINT_DIR 指向訓練產出的 checkpoint）
```
