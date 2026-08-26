# Mariogpt_Dou-expert 腳本說明

所有腳本從 `run/` 目錄執行。

---

## 腳本一覽

### 1. `path-expert_training.sh` — Path Expert 訓練

從頭訓練 Path Expert 模型（不載入現有 checkpoint）。

- 執行 `path/mario_gpt/diff_LR_train.py`
- 第 1 輪 LR = 5e-4，後續輪 LR = 1e-4
- 輸出至 `path/mario_gpt/M_multiround_round{N}/`

```bash
bash run/path-expert_training.sh
```

---

### 2. `structure-expert_training.sh` — Structure Expert 訓練

從頭訓練 Structure Expert 模型（不載入現有 checkpoint）。

- 執行 `structure/mario_gpt/diff_LR_train.py`
- 第 1 輪 LR = 5e-4，後續輪 LR = 1e-4
- 輸出至 `structure/mario_gpt/M_multiround_round{N}/`

```bash
bash run/structure-expert_training.sh
```

---

### 3. `run_pipeline.sh` — 雙專家生成 Pipeline

依序執行 Path Expert 生成 → Structure Expert 生成。

- Step 1：`path/mario_gpt/multi_test.py --count N`
  - 生成 N 個路徑地圖，輸出至 `path/mario_gpt/multi_test/`
- Step 2：`structure/mario_gpt/pipeline_pathexpert.py`
  - 讀取所有路徑檔，強制植入路徑 token，輸出至 `structure/mario_gpt/pipeline_out/`

```bash
bash run/run_pipeline.sh [COUNT]
```

| 參數 | 說明 | 預設值 |
|------|------|--------|
| COUNT | 生成次數 | 10 |

範例：
```bash
bash run/run_pipeline.sh       # 生成 10 個
bash run/run_pipeline.sh 20    # 生成 20 個
```

---

## 執行順序建議

```
1. 訓練 Path Expert      →  bash run/path-expert_training.sh
2. 訓練 Structure Expert →  bash run/structure-expert_training.sh
3. 執行生成 Pipeline     →  bash run/run_pipeline.sh [COUNT]
```

訓練完成後，記得在各自的 `multi_test.py` 和 `pipeline_pathexpert.py` 中將 `CHECKPOINT_DIR` 指向訓練產出的 checkpoint 路徑。
