# 1. 找 char -> tile 對應表 / 渲染字典
grep -rn "mm-MMM\|W-J-END\|R-BIG\|smb-question\|smb-breakable\|smb-tube\|tile_dict\|char2tile\|TILE_\|mapping" --include=*.py . | grep -iv pipeline_pathexpert | head -30

# 2. utils.py 渲染函式（char -> image 的對應就在這）
sed -n '100,175p' utils.py

# 3. data/tiles 完整檔名（檔名就是 glyph 線索）
ls data/tiles/

# 4. 任一張「真實訓練關卡」的字元集（確認 target 編碼）
#    之前沒找到訓練 .txt，改從 dataset.py 看它從哪讀：
sed -n '1,90p' dataset.py
