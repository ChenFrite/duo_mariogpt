##cd .../structure-expert/mario_gpt_structure
export MARIO_CKPT=Structure-GPT2-v2/iteration_10000
export MARIO_TOK=shyamsn97/Mario-GPT2-700-context-length
export PATH_DIR=../../path/mario_gpt_path/multi_test
export N_PATHS=10 N_PROMPTS=4 TARGET_COLS=16 SEG_COLS=2 K_LIST=4 N_TRIALS=1
export LAMBDA_S=1.0 LAMBDA_P=0.25 SEED=0
python -u batch_search_vs_greedy.py >> a5.log
