"""
Teacher-forced 檢測：checkpoint 有沒有學會「結構 tile」。

把 MarioDataset 的 [path | future path | structure] 樣本整段餵進模型，
只看 structure 段（loss_mask==1），分兩類統計 argmax 準確率與 CE：
  path/-  : 目標是路徑 token 或 '-'（相對容易，路徑可從 prefix 抄）
  STRUCT  : 目標是 X/S/?/Q/<>/[]/T/E/o... 等真正的結構 tile
STRUCT acc ≈ 0 代表模型只會吐路徑（例如舊的 path-only checkpoint）。

用法：
  python probe_ckpt.py                                  # 預設 Structure-GPT2-v2/iteration_10000，val split
  python probe_ckpt.py Structure-GPT2-v2/iteration_5000
  python probe_ckpt.py <ckpt> --split all --n 32
"""
import argparse
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F

from mario_gpt_structure import MarioLM, MarioDataset
from mario_gpt_structure.dataset import PATH_TOKENS

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
VAL_FRACTION = 0.1  # 與 train.py 相同：關卡最後 10% 的 patch 是 held-out val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?",
                    default=os.path.join(_THIS_DIR, "Structure-GPT2-v2", "iteration_10000"))
    ap.add_argument("--split", choices=["val", "all"], default="val",
                    help="val = 只抽 held-out patch（較客觀）；all = 整個 dataset")
    ap.add_argument("--n", type=int, default=16, help="抽樣 patch 數")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lm = MarioLM(lm_path=args.ckpt, tokenizer_path=TOKENIZER_PATH)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = lm.to(dev)
    lm.eval()
    tok = lm.tokenizer
    ds = MarioDataset(tok)

    pool = range(int(len(ds) * (1 - VAL_FRACTION)), len(ds)) if args.split == "val" else range(len(ds))
    random.seed(args.seed)
    idx = random.sample(list(pool), min(args.n, len(pool)))

    ids, mask, pos, attn = (t.to(dev) for t in ds[idx])
    enc = torch.stack([lm.prompter(ids[i][mask[i] == 1])[1] for i in range(len(idx))])
    enc = enc.view(len(idx), 1, -1).to(dev)

    with torch.no_grad():
        logits = lm.lm(input_ids=ids, attention_mask=attn,
                       encoder_hidden_states=enc, position_2d=pos).logits

    sl, lab, m = logits[:, :-1], ids[:, 1:], mask[:, 1:] == 1
    ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)).float(), lab.reshape(-1),
                         reduction="none").view_as(lab)
    pred = sl.argmax(-1)

    path_like = set(PATH_TOKENS) | {"-"}
    n, hit, ce_sum = Counter(), Counter(), Counter()
    struct_preds = Counter()
    for b in range(len(idx)):
        for t in torch.nonzero(m[b]).flatten().tolist():
            ch = tok.decode([lab[b, t].item()])
            k = "path/-" if ch in path_like else "STRUCT"
            n[k] += 1
            ce_sum[k] += ce[b, t].item()
            hit[k] += int(pred[b, t] == lab[b, t])
            if k == "STRUCT":
                struct_preds[tok.decode([pred[b, t].item()])] += 1

    print(f"checkpoint: {args.ckpt}")
    print(f"split={args.split}  patches={len(idx)}")
    for k in ("path/-", "STRUCT"):
        if n[k]:
            print(f"{k:7s} n={n[k]:5d} acc={hit[k] / n[k]:.3f} CE={ce_sum[k] / n[k]:.3f}")
    print(f"masked CE all: {((ce * m).sum() / m.sum()).item():.4f}")
    print("pred chars on STRUCT targets:", struct_preds.most_common(8))


if __name__ == "__main__":
    main()
