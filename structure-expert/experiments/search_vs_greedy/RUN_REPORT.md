# Phase 0B：指定 checkpoint 結構生成診斷

執行日期：2026-10-01（Asia/Taipei）

## 設定

- 模型與 tokenizer：mario_gpt_structure/Mario-GPT2-700-context-length_29/iteration_10000
- Python：/home/isslab411/miniconda3/envs/marioexpert/bin/python
- PyTorch 2.13.0+cu130；Transformers 4.30.2；RTX 4090；CUDA
- 路徑：mario_gpt_structure/test_path.txt
- STRUCT prompt：some blocks, low elevation
- seed=0；temperature=2.0；top_k=16；使用本機快取，未下載模型
- 完整 sampler 生成 14×90；評分範圍為前 16 欄；mock A*。

## 結果與阻斷

結構 token 為 0/1260。forced decoding 前的 1260 個模型採樣 token 亦只有空白與 PATH token；free 位置有 1015 個空白與 115 個額外 PATH token。指定 PATH 位置 mismatch=0。

這不是統計器漏算現有結構，也不是 forced decoding 把結構覆蓋掉；本次原始採樣本身沒有結構。模型可載入與完成生成，但尚不符合 Phase 0B 驗收條件。尚不能僅依此斷定 checkpoint 未訓練好，需進一步檢查權重與訓練時模型／tokenizer／位置編碼／輸入格式相容性。

前 16 欄 mock reward=0.492424；solvable=0；adh_struct=0.5；adh_path=0.484848。這不是實際 A* 通關結果。

依 PLAN.md 的全零結構阻斷條件，未執行 path 難度掃描。後續先核對 checkpoint 的訓練來源與當時模型實作，再做受控診斷；結構非全零且評分核對通過後，才進行掃描。

## 產物與腳本修正

- diagnose.log：模型載入、原始 token 頻次、完整地圖与分項 reward。
- diagnose_seed0/diagnosis.json：絕對模型／路徑、設定與結果。
- diagnose_seed0/sampler_baseline.txt、sampler_baseline.png：完整輸出。
- diagnose_sampler.py：補上 CUDA、Python/NumPy/PyTorch seed、實際傳入固定 STRUCT prompt、forced 前 token 追蹤、free-position 計數、JSON／PNG 保存；全零時退出碼為 2。

原腳本未將評分 target_prompt 傳入生成，會隨機抽生成 prompt；此次已修正，避免生成與評分條件不一致。
