cd ~/Downloads/mariogpt_two_expert/20260826_Mriogpt_Dou-expert/structure-expert/mario_gpt_structure

# 每次執行寫到新的 log（a4_MMDD_HHMMSS.log），不會覆蓋舊結果；畫面上照樣即時顯示
LOG="a4_$(date +%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "log: $LOG"
export MARIO_CKPT=Structure-GPT2-v2/iteration_10000
export MARIO_TOK=shyamsn97/Mario-GPT2-700-context-length

export PATH_FILE=test_path.txt
export TARGET_COLS=8 SEG_COLS=2 K=4 N_TRIALS=5


# 1) 加長到 16 欄（仍 ≤25，留在 block 0）
# 2) 固定一個「打弱可控屬性」的 prompt：enemy/rect/coin 是 greedy 最常失手的地方
# 3) 把 reward 權重移到 adh_struct（solvable 恆 1、adh_path 也飽和，留著只是常數）
export TARGET_COLS=16 SEG_COLS=2 K=6 N_TRIALS=10
export TARGET_PROMPT="no pipes, many enemies, many blocks, many rects, no coins, high elevation, "
export LAMBDA_S=1.0 LAMBDA_P=0.25
python -u compare_search_greedy.py

