import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


class RopeGPT2Attention(GPT2Attention):
    def __init__(self, config):
        super().__init__(config)
        self.head_dim = self.split_size // self.num_heads
        assert self.head_dim % 2 == 0

        inv_freq_x = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        inv_freq_y = 1.0 / ( 5000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("inv_freq_x", inv_freq_x)
        self.register_buffer("inv_freq_y", inv_freq_y)
        self._rope_position_2d = None

    def _apply_rope_2d(self, q, k, position_2d):
        device = q.device
        B, nH, T, Dh = q.shape
        assert Dh == self.head_dim

        x_pos = position_2d[..., 0].to(device=device, dtype=self.inv_freq_x.dtype)
        y_pos = position_2d[..., 1].to(device=device, dtype=self.inv_freq_y.dtype)

        freqs_x = torch.einsum("bt,d->btd", x_pos, self.inv_freq_x)
        freqs_y = torch.einsum("bt,d->btd", y_pos, self.inv_freq_y)

        cos_x = torch.cos(freqs_x).unsqueeze(1)
        sin_x = torch.sin(freqs_x).unsqueeze(1)
        cos_y = torch.cos(freqs_y).unsqueeze(1)
        sin_y = torch.sin(freqs_y).unsqueeze(1)

        qx, qy = q[..., :Dh//2], q[..., Dh//2:]
        kx, ky = k[..., :Dh//2], k[..., Dh//2:]

        qx_rot = (qx * cos_x) + (rotate_half(qx) * sin_x)
        qy_rot = (qy * cos_y) + (rotate_half(qy) * sin_y)
        kx_rot = (kx * cos_x) + (rotate_half(kx) * sin_x)
        ky_rot = (ky * cos_y) + (rotate_half(ky) * sin_y)

        return torch.cat([qx_rot, qy_rot], dim=-1), torch.cat([kx_rot, ky_rot], dim=-1)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        position_2d=None,
        **kwargs,
    ):
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        rope_pos = position_2d if position_2d is not None else self._rope_position_2d
        if rope_pos is not None:
            q_len = query.size(2)
            q_rope_pos = rope_pos[:, -q_len:, :]
            query, key = self._apply_rope_2d(query, key, q_rope_pos)

        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class GPT2With2DSinusoids(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        for i, block in enumerate(self.transformer.h):
            rope_attn = RopeGPT2Attention(config)
            rope_attn.load_state_dict(block.attn.state_dict(), strict=False)
            self.transformer.h[i].attn = rope_attn

        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_2d=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.transformer.wte(input_ids)

        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        extended_attention_mask = self.transformer.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        hidden_states = inputs_embeds
        presents = [] if use_cache else None

        for block in self.transformer.h:
            residual = hidden_states
            hidden_states = block.ln_1(hidden_states)
            block.attn._rope_position_2d = position_2d

            attn_outputs = block.attn(
                hidden_states,
                attention_mask=extended_attention_mask,
                use_cache=use_cache,
                **kwargs
            )
            attn_output = attn_outputs[0]
            if use_cache:
                presents.append(attn_outputs[1])

            hidden_states = residual + attn_output

            if encoder_hidden_states is not None and hasattr(block, 'crossattention'):
                residual = hidden_states
                hidden_states = block.ln_cross_attn(hidden_states)
                cross_attn_output = block.crossattention(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )[0]
                hidden_states = residual + cross_attn_output

            residual = hidden_states
            hidden_states = block.ln_2(hidden_states)
            hidden_states = residual + block.mlp(hidden_states)

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            return (logits, presents)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config, **kwargs)
