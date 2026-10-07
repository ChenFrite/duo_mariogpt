# 訓練 target 的真實字元集（終極裁決）
python -c "from mario_gpt_structure.level import FULL_LEVEL_STR_WITH_PATHS as s; print('uniq:', ''.join(sorted(set(s.replace(chr(10),''))))); print('rows:', s.count(chr(10))+1, 'len:', len(s))"

# 看它前 14 列長相（地面是 X 還是 M？有沒有 <> S ? o）
python -c "from mario_gpt_structure.level import FULL_LEVEL_STR_WITH_PATHS as s; print(chr(10).join(s.splitlines()[:14]))"
