from __future__ import annotations

import os,sys


from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from mario_gpt_path.lm.base import BaseMarioLM
from mario_gpt_path.prompter import Prompter
from mario_gpt_path.sampler import GPTSampler, SampleOutput
from mario_gpt_path.lm.gpt2_2dpos_model import GPT2With2DSinusoids

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

        from mario_gpt_path.lm.gpt2_2dpos_model import GPT2With2DSinusoids
        print(">>> After load, model class =", type(self.lm))
        assert isinstance(self.lm, GPT2With2DSinusoids), \
        f"Expected GPT2With2DSinusoids, got {type(self.lm)}"
    #batch_size: initial map used as start point of generation
    def generate_seed(self, length: int, batch_size: Optional[int] = None):
        """
        This line converts the string "X" into a token ID tensor:
        ## self.tokenizer("X", return_tensors="pt") - Tokenizes "X" and returns PyTorch tensors
        ##input_ids - Extracts the token IDs from the tokenizer output
        ##squeeze() - Removes dimensions of size 1, converting from shape (1, 1) to scalar tensor
        The result is a single token ID representing "X" that can be used as the starting token for level generation.
        """
        seed = self.tokenizer("-", return_tensors="pt").input_ids.squeeze()

        """
        If no batch_size, creates 1D tensor with length copies of "X" token
        """
        if batch_size is None:
            return seed.repeat(length)
        
        """
        If batch_size provided, creates 2D tensor of shape 
        (batch_size, length) where each row contains length "X" tokens
        """
        return seed.view(1, 1).repeat(batch_size, length)

    def load_pretrained_lm(self, path: str, lm_kwargs):
        if path == "random":
            print("Initializing random GPT2With2DSinusoids weights...")
            config = AutoConfig.from_pretrained(self.BASE_LM_PATH, **{**lm_kwargs, "add_cross_attention": True})
            model = GPT2With2DSinusoids.from_config(config)
            return model
        
        print(f"------ Loading pretrained model from {path} into GPT2With2DSinusoids")
        abs_path = os.path.abspath(path)
        config = AutoConfig.from_pretrained(abs_path, **{**lm_kwargs, "add_cross_attention": True})
        model = GPT2With2DSinusoids.from_pretrained(abs_path, config=config)
        return model
    
    def load_pretrained_tokenizer(
        self, path: str, tokenizer_kwargs: Dict[str, Any]
    ) -> GPT2Tokenizer:
        if path == "random":
            return AutoTokenizer.from_pretrained(
                self.BASE_TOKENIZER_PATH, **tokenizer_kwargs
            )
        print(f"------ Loading pretrained tokenizer from {path}")
        return AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

    def sample(
        self,
        seed: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        num_steps: int = 1,
        temperature: float = 2.0,
        encoder_hidden_states: torch.Tensor = None,
        use_tqdm: bool = False,
        return_tensor: bool = False,
        level_height: int = 14,
        positions_2d_NonExpand: Optional[torch.Tensor] = None,
    ) -> SampleOutput:
        sampler = GPTSampler(self, temperature, 16, self.context_len, use_tqdm, positions_2d_NonExpand=positions_2d_NonExpand)
        return sampler(
            seed=seed,
            prompts=prompts,
            num_steps=num_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_tensor=return_tensor,
            level_height=level_height,
        )
