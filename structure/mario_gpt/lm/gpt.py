from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mario_gpt.lm.base import BaseMarioLM
from mario_gpt.prompter import Prompter
from mario_gpt.sampler import GPTSampler, SampleOutput
from mario_gpt.lm.gpt2_2dpos_model import GPT2With2DSinusoids

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"


class MarioGPT(BaseMarioLM):
    PRETRAINED_LM_PATH = PRETRAINED_MODEL_PATH
    PRETRAINED_TOKENIZER_PATH = PRETRAINED_MODEL_PATH

    BASE_LM_PATH = "distilgpt2"
    BASE_TOKENIZER_PATH = "distilgpt2"

    def __init__(
        self,
        lm: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        context_len: int = 700,
        prompter: Optional[Prompter] = None,
        lm_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        lm_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(
            lm,
            tokenizer,
            context_len,
            lm_path,
            tokenizer_path,
            lm_kwargs,
            tokenizer_kwargs,
        )
        self.prompter = prompter
        if prompter is None:
            self.prompter = Prompter(self.tokenizer)

        from mario_gpt.lm.gpt2_2dpos_model import GPT2With2DSinusoids
        assert isinstance(self.lm, GPT2With2DSinusoids), \
            f"Expected GPT2With2DSinusoids, got {type(self.lm)}"

    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        seed = self.tokenizer("-", return_tensors="pt").input_ids.squeeze()
        if batch_size is None:
            return seed.repeat(length)
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs: Dict[str, Any]) -> GPT2Model:
        if path == "random":
            config = AutoConfig.from_pretrained(
                self.BASE_LM_PATH, **{**lm_kwargs, "add_cross_attention": True}
            )
            model = GPT2With2DSinusoids.from_config(config)
            return model
        config = AutoConfig.from_pretrained(
            path, **{**lm_kwargs, "add_cross_attention": True}
        )
        model = GPT2With2DSinusoids.from_pretrained(path, config=config)
        return model

    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> GPT2Tokenizer:
        if path == "random":
            return AutoTokenizer.from_pretrained(self.BASE_TOKENIZER_PATH, **tokenizer_kwargs)
        return AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

    def sample(
        self,
        path_grid: Optional[List[List[str]]] = None,
        prompts: Optional[List[str]] = None,
        num_cols: int = 100,
        temperature: float = 2.0,
        top_k: int = 16,
        use_tqdm: bool = False,
        level_height: int = 14,
    ) -> SampleOutput:
        sampler = GPTSampler(
            self,
            temperature=temperature,
            top_k=top_k,
            use_tqdm=use_tqdm,
        )
        return sampler.sample(
            path_grid=path_grid,
            prompts=prompts,
            num_cols=num_cols,
            level_height=level_height,
        )
