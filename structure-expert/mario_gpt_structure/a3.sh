# 1. round1 訓練到底有沒有收斂 / 跑了幾步（為何沒存權重）

echo '1: tail round1 訓練到底有沒有收斂 / 跑了幾步（為何沒存權重）'
tail -20 M_multiround_round1/loss_log.csv

echo '2: wc -l round1 訓練到底有沒有收斂 / 跑了幾步（為何沒存權重）'
wc -l M_multiround_round1/loss_log.csv

# 2. 訓練設定：SAVE_ITERATION / 總步數 / NUM_ROUNDS

echo '3: 訓練設定：SAVE_ITERATION / 總步數 / NUM_ROUNDS'
sed -n '18,45p' multi_round_train.py
