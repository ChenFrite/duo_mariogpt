from __future__ import annotations

import os,sys

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image
from tqdm import tqdm
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper

from mario_gpt_path.lm.base import BaseMarioLM
from mario_gpt_path.prompter import Prompter
from mario_gpt_path.utils import (
    convert_level_to_png,
    load_level,
    save_level,
    trim_level,
    view_level,
)

DEBUG = False

@dataclass
class SampleOutput:
    level: Optional[List[str]]
    prompt: Optional[str] = None
    img: Optional[Image] = None
    sample_predictions_str: Optional[List[str]] = None
    sample_predictions_img: Optional[Image] = None
    level_tensor: Optional[torch.Tensor] = None
    sample_predictions_tensor: Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        level_tensor: torch.Tensor,
        sample_predictions_tensor: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
    ) -> SampleOutput:
        # batch = 1
        level = None
        img = None

        try:
            level = view_level(level_tensor, tokenizer)
            img = convert_level_to_png(level)[0]
        except Exception as e:
            print(
                f"Failed to generate string or image representation for full level! Got error {e}"
            )
            level = None
            img = None
        try:
            sample_predictions_str = view_level(sample_predictions_tensor, tokenizer)
            sample_predictions_img = convert_level_to_png(sample_predictions_str)[0]
        except Exception as e:
            print(
                f"Failed to generate string or image representation for sampled predictions! Got error {e}"
            )
            sample_predictions_str = None
            sample_predictions_img = None

        prompt = None
        if prompter is not None:
            prompt = prompter(level_tensor)[0]

        return SampleOutput(
            level,
            prompt,
            img,
            sample_predictions_str,
            sample_predictions_img,
            level_tensor,
            sample_predictions_tensor,
        )

    @classmethod
    def from_level_predictions(
        cls,
        level: torch.Tensor,
        sample_predictions: torch.Tensor,
        tokenizer,
        prompter: Optional[Prompter] = None,
    ) -> Union[SampleOutput, List[SampleOutput]]:
        level_tensor = trim_level(level).squeeze().detach().cpu()
        sample_predictions_tensor = (
            trim_level(sample_predictions).squeeze().detach().cpu()
        )

        if len(level_tensor.shape) == 1:
            return SampleOutput.create(
                level_tensor, sample_predictions_tensor, tokenizer, prompter
            )

        out = []
        for _level_tensor, _sample_predictions_tensor in zip(
            level_tensor, sample_predictions_tensor
        ):
            sample_output = SampleOutput.create(
                _level_tensor, _sample_predictions_tensor, tokenizer, prompter
            )
            out.append(sample_output)
        print('111111111:', level)
        print('222222222:', out)
        return out

    def save0(self, filename: str) -> str:
        save_level(self.level, filename)
    def save(self, filename: str) -> str:
        """
        Save the generated level (self.level) to a text file with newline per row.
        """
        if self.level is None:
            raise ValueError("No level data to save.")

        with open(filename, "w") as f:
            for line in self.level:
                f.write(line + "\n")
        return filename

    @classmethod
    def load(cls, filename: str) -> SampleOutput:
        level = load_level(filename)
        return SampleOutput(level=level)


class GPTSampler:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        temperature: float = 2.0,
        top_k: int = 16,
        context_len: int = 700,
        use_tqdm: bool = False,
        use_argmax: bool = False,
        positions_2d_NonExpand: Optional[torch.Tensor] = None,
    ):
        self.mario_lm = mario_lm
        self.temperature = temperature
        self.top_k = top_k
        self.context_len = context_len
        self.use_tqdm = use_tqdm
        self.use_argmax = use_argmax
        self.logits_processor = LogitsProcessorList()
        self.logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k),  # number of characters
                TemperatureLogitsWarper(temperature),
            ]
        )
        self.positions_2d_NonExpand = positions_2d_NonExpand

    @property
    def device(self) -> torch.device:
        return self.mario_lm.device

    def step(
        self,
        seed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_2d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            attention_mask = torch.ones_like(seed).to(seed.device)
            input_ids = seed
            out = self.mario_lm.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
                position_2d=position_2d,  #用本回合對齊過的
            )
            logits = out.logits.detach()
            if len(logits.shape) == 2:
                logits = logits.view(1, 1, -1)
            next_token_logits = logits[:, -1, :]

            if self.use_argmax:
                next_tokens = next_token_logits.argmax(-1)
            else:
                next_token_scores = self.logits_processor(input_ids, next_token_logits)
                next_token_scores = self.logits_warper(input_ids, next_token_scores)
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return next_tokens, encoder_hidden_states
    
    def sample(
        self,
        seed: Union[Optional[torch.Tensor], Optional[SampleOutput]] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        encoder_hidden_states: torch.Tensor = None,
        return_tensor: bool = False,
        level_height: int = 14,
    ):
        self.mario_lm.eval()
        context_len = self.context_len - 28
        W = self.context_len // level_height
        H = level_height

        ##print(f'44444444444 self.context_len - 28={context_len}'), 

        with torch.no_grad():
            if seed is None:
                seed = self.mario_lm.generate_seed(1, batch_size=len(prompts)).to(
                    self.device
                )
                out_tensor = seed.to(self.device)
            elif isinstance(seed, SampleOutput):
                out_tensor = seed.level_tensor.to(self.device).squeeze()
            else:
                out_tensor = seed.to(self.device).squeeze()
            if len(out_tensor.shape) < 2:
                # if we pass in a single seed vector, then we repeat for each prompt
                # Otherwise, we treat inputs as separate seed-prompt pairs
                out_tensor = out_tensor.view(1, -1).repeat(len(prompts), 1)
            if encoder_hidden_states is None:
                if prompts is not None:
                    encoder_hidden_states = torch.stack(
                        [
                            self.mario_lm.prompter.output_hidden(prompt)
                            for prompt in prompts
                        ]
                    )
                else:
                    encoder_hidden_states = torch.stack(
                        [
                            self.mario_lm.prompter(sample_prompt=True)[1]
                            for _ in range(seed.shape[0])
                        ]
                    )
            encoder_hidden_states = encoder_hidden_states.to(
                self.device
            )  # b x 1 x hidden_dim
            encoder_hidden_states = encoder_hidden_states.view(
                out_tensor.shape[0], 1, -1
            )
            if not self.use_tqdm:
                bar = np.arange(num_steps)
            else:
                bar = tqdm(np.arange(num_steps))
            with torch.no_grad():
                batch_size=len(prompts)
                self.position_2d = self.positions_2d_NonExpand.unsqueeze(0).repeat(batch_size, 1, 1) #.to(device)
                for i in bar:
                    """
                    inp = out_tensor * 1
                    abs_len = inp.shape[-1]  # 目前累積的“絕對長度”（從關卡起點已生成的 token 數）

                    if len(out_tensor.shape) > 0 and out_tensor.shape[-1] > context_len:
                        diff = abs_len % level_height  # 讓視窗從列邊界對齊
                        ctx = context_len + diff
                        inp = inp[:, -ctx:] * 1
                        # 取出 "同一段 suffix" 的 2D 座標；self.position_2d 需是 [1, 700, 2] 或 [B, 700, 2]
                        # abs_len 是 out_tensor 的長度，suffix 起點是 abs_len-ctx
                        pos_slice = self.position_2d[:, abs_len-ctx:abs_len, :].to(self.device)
                    else:
                        ctx = abs_len
                        pos_slice = self.position_2d[:, :ctx, :].to(self.device)
                    """
                    abs_len = out_tensor.shape[-1]
                    height  = level_height          # 14
                    context_len = self.context_len  # 700
                    #why not minus 28?

                    if abs_len > context_len:
                        import math
                        end_col = math.ceil(abs_len/H)
                        start_col = end_col-W
                        start_token = start_col*H
                        end_token = min(abs_len,end_col*H)
                        ##start_token = abs_len - ctx
                        ##start_token = (start_token // H) * H  # 欄對齊
                        ##end_token   = start_token + ctx
                        inp         = out_tensor[:, start_token:end_token]
                        pos_slice = self.position_2d[:, :(end_token - start_token), :]
                        #start = abs_len - context_len
                        # 對齊到欄界（14 的倍數）
                        #start = (start // height) * height
                        #end   = start + context_len
                        #inp   = out_tensor[:, start:end]                 # [B,700]
                        #pos_slice = self.position_2d[:, :context_len, :] # [B,700,2] 固定用整張表
                    else:
                        inp   = out_tensor                               # [B,abs_len]
                        pos_slice = self.position_2d[:, :abs_len, :]     # 同步長度

                    next_tokens, encoder_hidden_states = self.step(
                        inp,
                        encoder_hidden_states=encoder_hidden_states,
                        position_2d=pos_slice,  # ← 傳入對齊長度的座標
                    )

                    out_tensor = torch.cat([out_tensor, next_tokens.unsqueeze(-1)], dim=-1)

                    if DEBUG and self.use_tqdm:
                        bar.set_description(
                            f"shape: {inp.shape}, {out_tensor.shape} first: {inp[0][0]}, last: {out_tensor[0][-1]}"
                        )
            if self.use_tqdm:
                bar.close()
        sample_out = SampleOutput.from_level_predictions(
            out_tensor,
            out_tensor[:, -num_steps:],
            self.mario_lm.tokenizer,
            self.mario_lm.prompter,
        )
        self.mario_lm.train()
        if return_tensor:
            return sample_out, out_tensor
        return sample_out

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


class BertSampler:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        temperature: float = 2.0,
        top_k: int = 16,
        context_len: int = 448,
        mask_proportion: float = 0.16,
    ):
        self.mario_lm = mario_lm
        self.temperature = temperature
        self.top_k = top_k
        self.logits_processor = LogitsProcessorList()
        self.logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(top_k),  # number of characters
                TemperatureLogitsWarper(temperature),
            ]
        )
        self.context_len = context_len
        self.mask_proportion = mask_proportion
        self.mask_portion = int(self.context_len * self.mask_proportion)
        self.mask_portion = self.mask_portion - self.mask_portion % 14 + 14

    @property
    def device(self) -> torch.device:
        return self.mario_lm.device

    def get_context(self, input_ids, mask_indices):
        start_idx = mask_indices[0]
        end_idx = mask_indices[-1]

        if input_ids.shape[-1] <= self.context_len:
            clipped = input_ids.shape[-1] % 14
            input_ids = input_ids[:clipped]

        portion = (self.context_len - self.mask_portion) / 2

        remainder = 0
        left = start_idx - portion
        if left < 0:
            remainder = -1 * left

        right = end_idx + portion + remainder

        return input_ids[left:right]

    def sample(
        self,
        seed: Union[torch.Tensor, SampleOutput],
        mask: torch.Tensor,
        return_tensor: bool = False,
    ):
        self.mario_lm.eval()
        mask_indices = mask.nonzero()
        input_ids = seed
        if isinstance(seed, SampleOutput):
            input_ids = seed.level_tensor.to(self.device).squeeze()

        input_id_list = []
        for i in range(input_ids.shape[0]):
            input_id = input_ids[i]
            mask_index = mask_indices[mask_indices[:, 0] == i][:, -1]
            input_id = self.get_context(input_id, mask_index)
            input_id_list.append(input_id)
        input_ids = torch.stack(input_ids, dim=0).to(self.device)

        attention_mask = torch.ones_like(input_ids).to(seed.device)

        if len(input_ids.shape) < 2:
            # if we pass in a single seed vector, then we repeat for each prompt
            # Otherwise, we treat inputs as separate seed-prompt pairs
            input_ids = input_ids.view(1, -1)

        out = self.mario_lm.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
        )
        logits = out.logits.detach()
        if len(logits.shape) == 2:
            logits = logits.view(1, 1, -1)

        if self.use_argmax:
            tokens = logits.argmax(-1)
        else:
            tokens_scores = self.logits_processor(input_ids, tokens)
            tokens_scores = self.logits_warper(input_ids, tokens_scores)
            probs = torch.nn.functional.softmax(tokens_scores, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        out = input_ids.detach()

        for i in range(input_ids.shape[0]):
            mask_index = mask_indices[mask_indices[:, 0] == i][:, -1]
            out[i, mask_index] = tokens[i, mask_index].detach()

        sample_out = SampleOutput.from_level_predictions(
            out,
            tokens,
            self.mario_lm.tokenizer,
            self.mario_lm.prompter,
        )
        self.mario_lm.train()
        if return_tensor:
            return sample_out, tokens
        return sample_out
