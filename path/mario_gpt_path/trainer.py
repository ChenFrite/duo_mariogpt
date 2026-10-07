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

from mario_gpt_path.dataset import MarioDataset
from mario_gpt_path.lm import BaseMarioLM, MarioLM
from mario_gpt_path.utils import (
    convert_level_to_png,
    chars_to_2d_level_column_first_bottom_to_top,
)


@dataclass
class TrainingConfig:
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    output_dir: str = "Mario-GPT2-700-context-length"
    learning_rate: float = 5e-4
    epsilon: float = 1e-9
    lr_warmup_steps: int = 100

    batch_size: int = 4
    total_steps: int = 1001
    mask_proportion: float = 0.0
    eval_iteration: int = 1000
    save_iteration: int = 1000
    save_prefix: int = 1000

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
    ):
        self.mario_lm = mario_lm
        self.train_dataset = train_dataset

        self.config = config or TrainingConfig()
        self.optimizer = optimizer or self.create_optimizer(self.config)
        self.lr_scheduler = lr_scheduler or self.create_lr_scheduler(
            self.config, self.optimizer
        )
        self.accelerator = self.create_accelerator(self.config)

    def prepare(self) -> Tuple[PreTrainedModel, Optimizer, Any]:
        from mario_gpt_path.lm.gpt2_2dpos_model import GPT2With2DSinusoids

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
        position_2d: Optional[torch.Tensor] = None,
        iteration: Optional[int] = -1,
    ):
        device = accelerator.device
        total_train_loss = 0
        indices = list(
            torch.randint(low=0, high=len(train_dataset), size=(batch_size,)).long()
        )

        batch = train_dataset[indices]
        b_input_ids = batch[0].view(batch_size, -1).to(device)
        b_labels = batch[0].view(batch_size, -1).to(device)
        attention_masks = batch[1].to(device)

        encoder_hidden_states = []
        for level in b_input_ids:
            prompt, encoder_hidden_state, _, _ = self.mario_lm.prompter(level)
            encoder_hidden_states.append(encoder_hidden_state)
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0).view(
            batch_size, 1, -1
        )

        with accelerator.accumulate(model):
            model.zero_grad()
            outputs = model(
                input_ids=b_input_ids,
                labels=b_labels,
                attention_mask=attention_masks,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
                position_2d=position_2d,
            )

            loss = outputs.loss
            logits = outputs.logits
            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            del loss, outputs, logits
            torch.cuda.empty_cache()

        return total_train_loss / batch_size, {}

    def eval_once(self, model, position_2d, batch_size=1):
        """在驗證集上計算 loss（這裡用 train_dataset 當 val set 示範）"""
        model.eval()
        device = self.accelerator.device

        indices = torch.randint(low=0, high=len(self.train_dataset), size=(batch_size,)).long()
        batch = self.train_dataset[indices]

        b_input_ids = batch[0].view(batch_size, -1).to(device)
        b_labels = batch[0].view(batch_size, -1).to(device)
        attention_masks = batch[1].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=b_input_ids,
                labels=b_labels,
                attention_mask=attention_masks,
                position_2d=position_2d,
            )
            val_loss = outputs.loss.item()

        model.train()
        return val_loss

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

        device = self.accelerator.device
        position_2d_full = self.train_dataset.positions_2d_full
        position_2d = position_2d_full.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        model, optimizer, lr_scheduler = self.prepare()

        # === 建立 CSV 檔案 ===
        os.makedirs(checkpoint_path, exist_ok=True)
        log_file = os.path.join(checkpoint_path, "loss_log.csv")
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "train_loss", "val_loss", "last_lr"])

        bar = tqdm(np.arange(total_steps))
        model.train()

        for i in bar:
            train_loss, _ = self.train_iter(
                self.accelerator,
                model,
                self.train_dataset,
                optimizer,
                lr_scheduler,
                batch_size,
                position_2d,
                i,
            )
            logs = {"loss": train_loss, "last_lr": lr_scheduler.get_last_lr()[0]}
            bar.set_description(f"{logs}")
            self.accelerator.log({**logs}, step=i)

            val_loss = ""
            if i > 0 and i % self.config.eval_iteration == 0:
                val_loss = self.eval_once(model, position_2d, batch_size)
                print(f"[Eval] step={i}, val_loss={val_loss:.4f}")

            # === 寫入 CSV ===
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([i, train_loss, val_loss, lr_scheduler.get_last_lr()[0]])

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