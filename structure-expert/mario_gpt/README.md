# Structure Expert — 腳本說明

---

## `train.py` — 單輪訓練（快速測試用）

從頭開始，跑一次固定步數的訓練。

- 起始模型：`BASE = "random"`
- Dataset：`MarioDataset(tokenizer)`（預設關卡資料集）
- 輸出至：`Mario-GPT2-700-context-length_29/`
- 預設：`total_steps=10001`，每 1000 步存一次

```python
# 設定區（可修改）
BASE       = "random"
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
- Dataset：`MarioDataset(tokenizer)`
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

## `pipeline_pathexpert.py` — 批次生成結構地圖（Pipeline 模式）

讀取 Path Expert 生成的路徑檔，將路徑 token 強制植入，生成對應的結構地圖。

- 從 `path/mario_gpt/multi_test/` 掃描所有 `generated_level_N.txt`
- 結構 prompt 隨機產生（pipe / enemy / block / coin / rect / elevation）
- 路徑 token（M/J/j/A/C/L/K/W/U/k/w/R/r）在生成時每步即時強制覆蓋
- 輸出：`pipeline_out/result_{N}.txt`、`result_{N}.png`、`report.txt`

```python
# 設定區（可修改）
CHECKPOINT_DIR  = "./M_multiround_round10/iteration_10000"
PATH_EXPERT_DIR = "../../path/mario_gpt/multi_test"
```

```bash
python pipeline_pathexpert.py
```

---

## `same_path_test.py` — 固定路徑批次生成（測試用）

載入一個固定的路徑檔（`test_path.txt`），對同一條路徑重複生成多次結構地圖，用於測試模型在相同路徑條件下的生成多樣性。

- 路徑來源：`test_path.txt`（固定，不從 Path Expert 掃描）
- 結構 prompt 每次隨機產生（pipe / enemy / block / coin / rect / elevation）
- 路徑 token 強制植入（與 pipeline_pathexpert.py 相同機制）
- 輸出：`multi_test/structure_{N}.txt`、`structure_{N}.png`、`prompt_check_report.txt`

```python
# 設定區（可修改）
GENERATE_COUNT = 100
CHECKPOINT_DIR = "./M_multiround_round10/iteration_10000"
PATH_FILE      = "./test_path.txt"
```

```bash
python same_path_test.py
```

---

## 建議使用順序

```
1. 訓練模型      →  python diff_LR_train.py
2. 執行 Pipeline →  python pipeline_pathexpert.py
                   （需先確認 Path Expert 已生成路徑檔）
                   （記得將 CHECKPOINT_DIR 指向訓練產出的 checkpoint）
```
