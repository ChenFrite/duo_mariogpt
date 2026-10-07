# Path → Structure 搜尋實驗計畫

更新日期：2026-10-01

## 目標與範圍

在已驗證可用的 Structure Expert 與乾淨路徑上，比較貪婪生成和一步前瞻搜尋，量化 reward 增益及計算成本，找出搜尋價值較大的路徑條件。搜尋優於貪婪是待驗證假設；零增益或負增益也須完整報告。

預設採用現有 checkpoint、短關卡（≤16 欄）、Phase 0–3。多輪訓練是獨立實驗；本計畫不等待它完成，也不包含長關卡 value model vϕ、完整 Gumbel 搜尋與蒸餾。

## 已確認的地基與待驗證事項

- pipeline_pathexpert.py 指向 mario_gpt_structure/M_multiround_round10/iteration_10000，但該目錄不存在。
- 現有候選模型為 mario_gpt_structure/Mario-GPT2-700-context-length_29/iteration_10000；存在不等於已驗證可用，更不能直接認定可達 reward ≈2.0。
- pipeline 路徑來源為 ../path/mario_gpt_path/multi_test，已有 generated_level_0.txt–generated_level_9.txt 共 10 個。是否 OOD 尚未驗證。
- pipeline_out/report.txt 的 10 個結果，其五種結構物件統計皆為 0、高度皆為 low；須檢查實際 tile、統計器與模型相容性，才能判定原因。
- 多輪訓練只有 round1/loss_log.csv，最後有效紀錄為 step 141，尾端有 252 bytes NUL，無 checkpoint，未發現訓練程序。中斷原因未知。
- scan_paths.py 目前位於 structure-expert 根目錄，但使用同層 lm、sampler 等匯入，須先修正匯入或移至 mario_gpt_structure 再執行。
- scan_paths.py 已提供貪婪與一步前瞻搜尋，但 K=4 固定、A* 使用 mock、僅輸出摘要，尚無固定 seed、逐 trial 分項、成本與 CI。
- reward 預設為 solvable + 0.5 × adh_struct + 0.5 × adh_path；先確認實際權重與各分項範圍，再解讀上限 2.0。

## Phase 0：地基確認

預估：唯讀檢查約 10 分鐘；模型 smoke test 另計。

### 0A：唯讀檢查（不跑模型）

在 structure-expert 下執行：

```bash
cd ~/Downloads/mariogpt_two_expert/20260826_Mriogpt_Dou-expert/structure-expert
rg -n 'CHECKPOINT|CKPT|lm_path|iteration|PATH_EXPERT_DIR|PATH_FILE|generated_level' mario_gpt_structure/pipeline_pathexpert.py
ls -la mario_gpt_structure/M_multiround_round*/
ls -la mario_gpt_structure/Mario-GPT2-700-context-length_29/iteration_10000/
rg -n 'MARIO_CKPT|MARIO_TOK|N_TRIALS|TARGET_COLS|MULTI_DIR|use_mock_astar' scan_paths.py
```

確認 checkpoint 含模型設定、權重、tokenizer 與自訂 2D 位置模型所需資訊；記錄模型路徑、檔案 hash、程式版本及套件版本。檢查 test_path.txt 和候選路徑的 14 列格式、各列等寬、合法 token、無 prompt 尾段，以及足夠的 target/future context。人工構造路徑須另標 synthetic，不能直接稱為分布內。

### 0A 檢查紀錄與歷史 reward 交叉確認

2026-10-01 已重新執行上述 pipeline 與目錄檢查：pipeline 仍指向不存在的 round10/iteration_10000；多輪輸出只有 round1/loss_log.csv。_29/iteration_10000 內有 config.json、pytorch_model.bin（538,820,679 bytes）、tokenizer.json、tokenizer_config.json、special_tokens_map.json、vocab.json 與 merges.txt。檔案存在僅證明具備載入候選，不代表完整性、相容性或生成品質已通過驗收。

「先前 reward 2.0 使用 round10」目前沒有可追溯證據。需找當時命令、執行 log、model/tokenizer 絕對路徑、工作目錄、hostname／GPU、程式版本與產物；區分模型曾移動／刪除、實際使用 _29、以及 411／5090 等不同機器環境。當前 ls 只能確認本機現況，不能判定歷史使用哪個模型。若無紀錄，標為歷史來源未知，以新的可復現 smoke test 驗證候選模型。

### 0B：模型與 reward smoke test

修正 scan_paths.py 的執行位置／匯入，明確指定 checkpoint 和匹配的 tokenizer。先用固定 path、固定 prompt、固定 seed 跑 8 欄，保存 TXT、PNG 與 reward 分項。核對強制路徑位置、自由位置的結構 token、統計器計數與人工檢查是否一致。

全零結構是阻斷點。smoke test 必須同時列印並保存原始 14 列 tile、全 token 頻次、非 forced 位置的 token 頻次、統計器識別的 tile 集合與逐項計數；把 path token 與結構 token 分開核對。以手工已知含結構的關卡校驗統計器，並追蹤 raw model token → decode → forced 覆蓋 → 儲存關卡的各階段，避免把生成或解碼問題直接歸因於訓練。

診斷與處理：

- 原始 tile 有結構，但統計為 0：檢查 tile 對應／計數規則，修正統計器後重算舊產物；舊報告可能是假警報。
- 原始 tile 只有 path＋空白：檢查 checkpoint 載入、tokenizer、decode、forced mask、資料與 prompt 條件；若管線正確，再評估模型／資料問題。
- checkpoint 未訓練充分或載入錯誤：以訓練紀錄、模型設定與受控對照建立證據，換可靠 checkpoint 或另行完成訓練。單次空白輸出不足以區分訓練不足與其他管線問題。

產出須記錄診斷類別、證據、修正與重測結果。未解釋原因前，不進 Phase 1。

通過條件：模型可載入並完成生成；tokenizer 與權重相容；路徑格式有效；reward 分項與實際輸出一致；沒有未解釋的全零結構。reward ≈2.0 僅作既有說法的復現目標，不作唯一驗收門檻。若失敗，先定位 checkpoint／資料／生成／評分問題，不進入搜尋規模化比較。

產出：模型與路徑 manifest、smoke test 產物、地基驗收紀錄。

## Phase 1：路徑難度掃描

預估半天，依 GPU 與 rollout 成本調整。

先以 N_TRIALS=1 掃描最多 20 個路徑；目前只有 10 個已確認的 Path Expert 輸出，不足部分須另準備與驗證。另加入乾淨 test_path 作對照。固定模型、prompt、temperature、top_k、seg_cols 與 reward 權重。

修正 scan_paths.py 匯入後，於 mario_gpt_structure 執行以下命令；假定腳本已放在該目錄。若只修正匯入並留在根目錄，改用 python ../scan_paths.py：

```bash
cd mario_gpt_structure
MARIO_CKPT="$PWD/Mario-GPT2-700-context-length_29/iteration_10000" MARIO_TOK="$PWD/Mario-GPT2-700-context-length_29/iteration_10000" MULTI_DIR="$PWD/../../path/mario_gpt_path/multi_test" N_TRIALS=1 TARGET_COLS=16 python scan_paths.py
```

此命令只能在 Phase 0 驗收通過後執行；模型路徑必須替換為最終驗證版本。test_path.txt 需明確納入輸入清單，因現有 scanner 只讀 generated_level_*.txt。

記錄 jump_ratio、greedy/search reward、三個分項、propose 次數、模型 token 數與 wall time。現有 jump_ratio 定義為 (J+j)/(M+J+j+A)，不是所有 path token 的跳躍比例；保存分子、分母，並以實際生成區段的比例為主、全路徑比例為輔。它只是難度代理，另記錄路徑長度與跳躍配置。

執行前必須修改 scan_paths.py：以本次生成的起始欄及 target_cols 切出實際生成區段，再計算 jump_ratio_target；另保留 jump_ratio_full 與各自分子、分母。若從第 0 欄生成，區段為 [0, target_cols)。確認切片索引與 env 輸出對齊，分母為 0 時標記缺乏代理資訊；只保留目前全路徑比例不符合驗收條件。

探索性分類：

- 飽和：greedy 接近 2.0 且 gain 接近 0。
- 低分：greedy <1.0；須看分項與輸出原因，不能直接判定 OOD。
- 甜蜜點候選：gain >0.1 且 greedy >1.0。

N_TRIALS=1 只用於篩選。若全飽和或全低分，先檢查評分，再調 8／12／16 欄或構造中等難度路徑；不得只挑有利 seed。

產出：逐 trial CSV、jump_ratio vs gain 表／圖、Phase 2 候選清單及選擇理由。

## Phase 2：搜尋與貪婪正式比較

預估 1–2 天，以 pilot 成本估計後決定總工作量。

先跑單一 cell pilot：target_cols=12、K=4、seg_cols=2、1 個甜蜜點候選 path、10 個預先固定的新 seed，比較貪婪、一步前瞻與 best-of-N。若尚無甜蜜點，使用有代表性的中等難度 path，明確標示未確認甜蜜點。量測三方法的總時間、逐 trial 時間、propose/model token/reward evaluation 成本與 peak GPU memory，尤其記錄搜尋的完整後續 rollout 成本。

pilot 完成後，依實測時間推估每個新增 cell 及總預算；不同 K／長度的成本不假定線性。先固定可負擔的主比較，再逐軸增加長度或 K；不要直接啟動完整 3×3 矩陣。資源仍不足則縮減 path／設定數，保留每格至少 10 seeds 與三方法比較，記錄縮減理由。pilot 可列入正式結果的條件是設定與分析規則未因其結果修改，否則只作探索資料。

擴展候選設定：每格至少 10 個預先列出的 seed，target_cols ∈ {8,12,16}，K ∈ {2,4,8}，seg_cols=2。先固定一個主要 prompt，再用額外 prompt 驗證結論是否穩健。保留容易、甜蜜點與較難路徑；Phase 1 的選擇偏差需揭露，正式使用新 seed，最好增加未參與篩選的路徑。

比較方法：

1. 貪婪基線：每步提出 K 個候選，選 logprob 最大者。這是候選集合內的貪婪，不等同逐 token argmax。
2. 一步前瞻：每步對 K 個候選各自 rollout 至終端，選終端 reward 最大者，再推進一段；此處仍包含完整後續 rollout 成本。
3. 成本對齊的 best-of-N 基線（本階段正式採用）：使用相同候選生成器與貪婪 rollout，生成 N 個完整關卡，以相同 mock evaluator 選最高 reward。N 依 pilot 的模型 token 評估成本預先決定，同一 cell 固定，不依測試 reward 自適應加樣本；保存 N、完整成本及與搜尋預算的殘餘差距，wall time 另報。評分失敗、timeout 與有效候選數必須保留，沒有有效輸出時記為失敗／未知，不靜默排除。

本階段不實作 rejection sampling。論文及圖表統一稱 best-of-N；若 proposal 原基線 (b) 要求 rejection sampling，應明示本實驗使用替代基線，不能聲稱已完成原基線。真正 rejection sampling 留待另立設定，預先定義接受門檻、最大嘗試次數及失敗規則。

先補足 runner：K、seed、prompt、A* mode、輸出目錄參數化；保存最終關卡及逐 trial 分項；對所有 rollout 的 propose/model token/reward evaluation 計數。共享初始候選與獨立 RNG stream 以減少方法間隨機消耗差異；seed 相同本身不保證候選相同。

公平條件：同 checkpoint、路徑、prompt、長度、proposal 設定與 reward。固定成本上限及 timeout，記錄 GPU、wall time、A* 呼叫與失敗；timeout／錯誤不得靜默丟棄。

統計：報告各方法平均 reward、配對 gain、95% bootstrap CI、solvable 率、兩種 adherence、成本倍數與 reward–cost 曲線。多路徑總結採路徑層級／階層 bootstrap；10 trials 是最低數量，若區間過寬依預先規則加樣本。CI 不支持正增益時，結論應為證據不足或無明顯提升。

產出：完整結果 CSV、關卡產物、gain ±95% CI at Nx cost、跨難度曲線，以及與成本對齊 best-of-N 的比較。

## Phase 3：真實 A* 驗證

預估半天起，依樣本與 timeout 調整。

先固定分層抽樣名單，涵蓋每種方法、難度、長度及成功／失敗例。用真實 A* 評估 Phase 2 保存的同一批最終關卡，避免重新生成引入隨機差異。test_env.py 的 REAL_ASTAR=1 目前只檢查該測試腳本的輸出，不能取代 Phase 2 產物的批次驗證；需新增或擴充 evaluator。

保存 A* 版本、執行時間限制、通關結果與 timeout／錯誤；後兩者列為未知，不直接當不通關。報告 mock／真實 A* 混淆矩陣、一致率與各方法真實 solvable 率，用真實 solvable 替換 reward 分項後重算配對 gain 與 CI。

此步只驗證最終輸出的可通關性與重評分。若要聲稱「真 A* 引導搜尋」有效，必須另外以真 A* 作搜尋中的 rollout evaluator；事後重評分不能支持這個聲稱。既有 mock／真實一致性仍須提供可追溯紀錄，不能預設成立。

論文措辭規則：若搜尋 rollout 使用 mock，只能在證據支持時聲稱「mock-guided search 的增益」與「最終結果經真實 A* 事後驗證仍支持增益」。摘要、圖例、表格、方法與結論都須一致，不使用「A*-guided search」或「真 A* 引導搜尋」描述本階段。真 A* 在每個 rollout 內評分是另一項高成本實驗，留待下一階段；單次耗時由本機實測，不預設固定 5 秒。

通過條件：真實 A* 下的效果方向與不確定性支持所聲稱的結論；若不一致，先修正 mock 或改用真 A*，重新執行必要比較。

## 決策與停止規則

- Checkpoint：優先驗證現有 _29/iteration_10000；若不可用，先修復／取得可靠模型，不以缺失的 round10 路徑繼續。
- 成本：預設先做 ≤16 欄，pilot 後設定每格與總運算上限；超限先縮小矩陣並記錄理由。>25 欄與 vϕ 留待下一階段。
- 論文定位：先完成搜尋有效性及成本的實證；完整 Gumbel＋蒸餾另立 Phase 4+。
- Phase 0 未通過、計數器未校驗、path 格式不合法時，停止後續規模化實驗。
- 未找到甜蜜點亦是有效結果，不以反覆改 seed 取代報告。

## 最小可行交付

Phase 0A 檔案與歷史來源核對 → Phase 0B 原始 tile／計數器診斷 → Phase 1 生成區段難度掃描 → Phase 2 單 cell pilot（12 欄、K=4、1 path、10 seeds）→ 依成本擴展正式比較 → Phase 3 真實 A* 抽驗。

建議產物目錄 experiments/search_vs_greedy/，保存 manifest、configs、逐 trial CSV、levels、logs、figures 與結果報告。實際執行前補足 scanner／runner／evaluator 的上述介面；本 PLAN.md 的建立不代表已執行實驗或已驗證搜尋增益。
