import os, sys

import os
import csv
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

DEBUG = False  # True

torch.cuda.empty_cache()

from accelerate import Accelerator
from PIL import ImageDraw
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup

from mario_gpt_structure.dataset import MarioDataset
from mario_gpt_structure.lm import BaseMarioLM, MarioLM
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    output_dir: str = "Mario-GPT2-700-context-length"
    learning_rate: float = 5e-4
    epsilon: float = 1e-9
    lr_warmup_steps: int = 150

    batch_size: int = 4
    total_steps: int = 10001
    mask_proportion: float = 0.0
    eval_iteration: int = 1000
    save_iteration: int = 1000
    save_prefix: int = 10000
    eval_batches: int = 8          # val batches averaged per eval

    def pretty_print(self):
        print("================== Training Config ==================")
        d = asdict(self)
        for k in d:
            print(f"{k} -- {d[k]}")
        print("================== MarioLM ==================")


class MarioGPTTrainer:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        train_dataset: MarioDataset,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        train_indices: Optional[list] = None,
        val_indices: Optional[list] = None,
    ):
        self.mario_lm = mario_lm
        self.train_dataset = train_dataset
        # held-out split (indices into train_dataset); default = everything / no val
        self.train_indices = list(train_indices) if train_indices is not None else list(range(len(train_dataset)))
        self.val_indices = list(val_indices) if val_indices is not None else []
        self._prompt_cache: dict = {}  # dataset idx -> BART prompt embedding (deterministic per sample)

        self.config = config or TrainingConfig()
        self.optimizer = optimizer or self.create_optimizer(self.config)
        self.lr_scheduler = lr_scheduler or self.create_lr_scheduler(
            self.config, self.optimizer
        )
        self.accelerator = self.create_accelerator(self.config)

    def prepare(self) -> Tuple[PreTrainedModel, Optimizer, Any]:
        from mario_gpt_structure.lm.gpt2_2dpos_model import GPT2With2DSinusoids

        print(">>> Trainer: After load, model class =", type(self.mario_lm.lm))
        assert isinstance(
            self.mario_lm.lm, GPT2With2DSinusoids
        ), f"Expected GPT2With2DSinusoids, got {type(self.mario_lm.lm)}"

        return self.accelerator.prepare(
            self.mario_lm.lm, self.optimizer, self.lr_scheduler
        )

    def create_optimizer(self, config: Any) -> Optimizer:
        params = self.mario_lm.lm.parameters()
        return AdamW(params, lr=config.learning_rate, eps=config.epsilon)

    def create_lr_scheduler(self, config: Any, optimizer: Optimizer) -> Any:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.total_steps,
        )

    def create_accelerator(self, config: Any) -> Accelerator:
        return Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=config.output_dir,
        )

    def unwrap(self) -> BaseMarioLM:
        return MarioLM(
            lm=self.accelerator.unwrap(self.mario_lm.lm),
            tokenizer=self.mario_lm.tokenizer,
            context_len=self.mario_lm.context_len,
            prompter=self.mario_lm.prompter,
        )

    def train_iter(
        self,
        accelerator: Accelerator,
        model: PreTrainedModel,
        train_dataset: MarioDataset,
        optimizer: Any,
        scheduler: Any,
        batch_size: int = 4,
        iteration: Optional[int] = -1,
        indices: Optional[list] = None,
    ):
        device = accelerator.device
        if indices is None:
            indices = [int(i) for i in np.random.choice(self.train_indices, size=batch_size, replace=False)]

        batch = train_dataset[indices]
        b_input_ids = batch[0].view(batch_size, -1).to(device)   # (B, MAX_SEQ_LEN)
        loss_mask   = batch[1].view(batch_size, -1).to(device)   # (B, MAX_SEQ_LEN)
        pos_2d      = batch[2].to(device)                         # (B, MAX_SEQ_LEN, 2)
        attn_mask   = batch[3].view(batch_size, -1).to(device)   # (B, MAX_SEQ_LEN)

        encoder_hidden_states = self._encoder_hidden_states(indices, b_input_ids, loss_mask)

        with accelerator.accumulate(model):
            model.zero_grad()
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
                position_2d=pos_2d,
            )

            logits = outputs.logits  # (B, 4H, vocab)

            # Masked cross-entropy: only the tile_curr segment (last H tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_input_ids[:, 1:].contiguous()
            shift_mask   = loss_mask[:, 1:].float().contiguous()

            loss_per_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none',
            ).view(batch_size, -1)

            loss = (loss_per_tok * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
            batch_loss = loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            del loss, outputs, logits
            torch.cuda.empty_cache()

        return batch_loss, {}

    def _encoder_hidden_states(self, indices, b_input_ids, loss_mask) -> torch.Tensor:
        """Prompt embedding per sample, computed from its structure segment (loss==1) and cached."""
        hs = []
        for i, idx in enumerate(indices):
            key = int(idx)
            if key not in self._prompt_cache:
                tile_mask = (loss_mask[i] == 1)
                level = b_input_ids[i][tile_mask] if tile_mask.any() else b_input_ids[i]
                _, h, _, _ = self.mario_lm.prompter(level)
                self._prompt_cache[key] = h.detach()
            hs.append(self._prompt_cache[key].to(b_input_ids.device))
        return torch.stack(hs, dim=0).view(len(indices), 1, -1)

    def eval_once(self, model, batch_size=1):
        """Masked CE on held-out val_indices, with the same prompt conditioning as training."""
        if not self.val_indices:
            return ""
        model.eval()
        device = self.accelerator.device
        rng = np.random.default_rng(0)  # same val batches every eval -> comparable curve
        total, count = 0.0, 0.0
        for _ in range(self.config.eval_batches):
            indices = [int(i) for i in rng.choice(self.val_indices, size=batch_size, replace=False)]
            batch = self.train_dataset[indices]

            b_input_ids = batch[0].view(batch_size, -1).to(device)
            loss_mask   = batch[1].view(batch_size, -1).to(device)
            pos_2d      = batch[2].to(device)
            attn_mask   = batch[3].view(batch_size, -1).to(device)
            encoder_hidden_states = self._encoder_hidden_states(indices, b_input_ids, loss_mask)

            with torch.no_grad():
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=attn_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    position_2d=pos_2d,
                )
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = b_input_ids[:, 1:].contiguous()
                shift_mask   = loss_mask[:, 1:].float().contiguous()
                loss_per_tok = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                    reduction='none',
                ).view(batch_size, -1)
                total += (loss_per_tok * shift_mask).sum().item()
                count += shift_mask.sum().item()

        model.train()
        return total / max(count, 1.0)

    def train(
        self,
        total_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if total_steps is None:
            total_steps = self.config.total_steps
        if batch_size is None:
            batch_size = self.config.batch_size

        self.accelerator.init_trackers("mario-gpt")

        checkpoint_path = self.config.output_dir
        logdir = os.path.abspath(self.accelerator.logging_dir)

        print(f"------- Config.save_iteration : {self.config.save_iteration}")
        if getattr(self.config, "pretty_print", None) is not None:
            self.config.pretty_print()

        model, optimizer, lr_scheduler = self.prepare()

        # === 建立 CSV 檔案 ===
        os.makedirs(checkpoint_path, exist_ok=True)
        log_file = os.path.join(checkpoint_path, "loss_log.csv")
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "train_loss", "train_loss_ma100", "val_loss", "last_lr"])

        MA_WINDOW = 100
        loss_history = []

        import random as _random
        index_pool: list = []

        bar = tqdm(np.arange(total_steps))
        model.train()

        for i in bar:
            if len(index_pool) < batch_size:
                new_indices = list(self.train_indices)
                _random.shuffle(new_indices)
                index_pool.extend(new_indices)
            indices = index_pool[:batch_size]
            index_pool = index_pool[batch_size:]

            train_loss, _ = self.train_iter(
                self.accelerator,
                model,
                self.train_dataset,
                optimizer,
                lr_scheduler,
                batch_size,
                i,
                indices=indices,
            )

            loss_history.append(train_loss)
            if len(loss_history) > MA_WINDOW:
                loss_history.pop(0)
            train_loss_ma = sum(loss_history) / len(loss_history)

            logs = {"loss": train_loss, "loss_ma100": train_loss_ma, "last_lr": lr_scheduler.get_last_lr()[0]}
            bar.set_description(f"{logs}")
            self.accelerator.log({**logs}, step=i)

            val_loss = ""
            if i > 0 and i % self.config.eval_iteration == 0:
                val_loss = self.eval_once(model, batch_size)
                print(f"[Eval] step={i}, val_loss={val_loss:.4f}")

            # === 寫入 CSV ===
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([i, train_loss, train_loss_ma, val_loss, lr_scheduler.get_last_lr()[0]])

            if i > 0 and i % self.config.save_iteration == 0:
                # 存模型
                self.mario_lm.save_model(checkpoint_path, i)
                print(f"----- Model saved at iteration {checkpoint_path},{i}")
                """
                # === 生成樣本關卡 (txt + png + TensorBoard) ===
                prompts = [
                    ("no rects, some blocks, high elevation, ", "1_no"),
                    ("many rects, some blocks, high elevation", "2_many"),
                    ("some rects, some blocks, high elevation", "3_some"),
                ]

                for prompt, tag in prompts:
                    generated_level = self.mario_lm.sample(
                        prompts=[prompt],
                        num_steps=1400,
                        temperature=2.0,
                        use_tqdm=True,
                        positions_2d_NonExpand=position_2d_full,
                    )
                    txt_name = f"{tag}_generated_level{i}.txt"
                    png_name = f"{tag}_generated_level{i}.png"

                    # 存 txt
                    generated_level.save(txt_name)
                    print(f"Saved {txt_name}")

                    # 存 png
                    try:
                        png_img, _, _ = convert_level_to_png(
                            generated_level.level_tensor, self.mario_lm.tokenizer
                        )
                        png_img.save(png_name)
                        print(f"Saved {png_name}")

                        # TensorBoard log
                        tracker = self.accelerator.get_tracker("tensorboard")
                        if hasattr(tracker, "writer"):
                            tracker.writer.add_image(
                                f"sample/{tag}", np.array(png_img), i, dataformats="HWC"
                            )
                    except Exception as e:
                        print(f"Failed to save PNG/TensorBoard: {e}")

                    del generated_level
                torch.cuda.empty_cache()
                """