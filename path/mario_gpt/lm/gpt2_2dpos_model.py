import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

# rope_gpt2_2d.py
import math
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


def rotate_half(x):
    # x[..., 0::2], x[..., 1::2] 交錯為 (x1, x2)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # [-x2, x1]
    x_rot = torch.stack((-x2, x1), dim=-1)  # [..., D/2, 2]
    return x_rot.flatten(-2)                # [..., D]

class RopeGPT2Attention(GPT2Attention):
    """
    在 HuggingFace GPT2Attention 基礎上插入 RoPE 2D。
    用法：把 GPT2Model.transformer.h[i].attn 替換成本類別的實例。
    在 forward 前把 position_2d 掛到 self._rope_position_2d 供使用。
    """
    def __init__(self, config):
        super().__init__(config)

        # 每 head 的維度（HF 內部叫 head_dim）
        self.head_dim = self.split_size // self.num_heads
        assert self.head_dim % 2 == 0, "head_dim 必須是偶數"

        # 分別為 X/Y 設不同頻率底數（可自行調）
        inv_freq_x = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        inv_freq_y = 1.0 / ( 5000.0 ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        self.register_buffer("inv_freq_x", inv_freq_x)  # [Dh/2]
        self.register_buffer("inv_freq_y", inv_freq_y)  # [Dh/2]

        # 供外部在每次 forward 前注入：shape [B, T, 2] (x, y)
        self._rope_position_2d = None

    def _apply_rope_2d(self, q, k, position_2d):
        """
        q, k: [B, nH, T, Dh]
        position_2d: [B, T, 2]，整個 batch 共用；或你也可以做每樣本不同
        """
        device = q.device
        B, nH, T, Dh = q.shape
        assert Dh == self.head_dim

        # 取 x,y（long → float），並擴成 [B, 1, T, Dh/2] 的角頻率
        x_pos = position_2d[..., 0].to(device=device, dtype=self.inv_freq_x.dtype)  # [B, T]
        y_pos = position_2d[..., 1].to(device=device, dtype=self.inv_freq_y.dtype)  # [B, T]

        # [T, Dh/2] ← [T,1] * [1,Dh/2]（這裡用 outer/einsum）
        freqs_x = torch.einsum("bt,d->btd", x_pos, self.inv_freq_x)  # [B,T,Dh/2]
        freqs_y = torch.einsum("bt,d->btd", y_pos, self.inv_freq_y)  # [B,T,Dh/2]

        # 擴成 [B,1,T,Dh/2] 以便 broadcast 到 nH
        cos_x = torch.cos(freqs_x).unsqueeze(1)  # [B,1,T,Dh/2]
        sin_x = torch.sin(freqs_x).unsqueeze(1)  # [B,1,T,Dh/2]
        cos_y = torch.cos(freqs_y).unsqueeze(1)  # [B,1,T,Dh/2]
        sin_y = torch.sin(freqs_y).unsqueeze(1)  # [B,1,T,Dh/2]

        # 把 head_dim 二等分：前半專給 X、後半專給 Y
        qx, qy = q[..., :Dh//2], q[..., Dh//2:]  # [B,nH,T,Dh/2]
        kx, ky = k[..., :Dh//2], k[..., Dh//2:]

        # 標準 RoPE：x*cos + rotate_half(x)*sin
        qx_rot = (qx * cos_x) + (rotate_half(qx) * sin_x)
        qy_rot = (qy * cos_y) + (rotate_half(qy) * sin_y)
        kx_rot = (kx * cos_x) + (rotate_half(kx) * sin_x)
        ky_rot = (ky * cos_y) + (rotate_half(ky) * sin_y)

        q_rot = torch.cat([qx_rot, qy_rot], dim=-1)
        k_rot = torch.cat([kx_rot, ky_rot], dim=-1)
        return q_rot, k_rot

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # === 以下基本照抄 HF GPT2Attention.forward ===
        # c_attn: Linear，輸出 [B, T, 3*embed]; split 成 q,k,v
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        # 變形為 multi-head: [B, nH, T, Dh]
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key   = self._split_heads(key,   self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # === 插入 RoPE-2D（若有 position_2d）===
        if self._rope_position_2d is not None:
            position_2d = self._rope_position_2d
            # 若 seq_len 不一致（例如做過滑窗），把 position_2d 切齊
            if position_2d.size(1) != query.size(2):
                position_2d = position_2d[:, :query.size(2), :]
            query, key = self._apply_rope_2d(query, key, position_2d)

        # past kv cache
        if layer_past is not None:
            past_key, past_value = layer_past
            key   = torch.cat((past_key,   key),   dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        # 注意力（含 causal + attention_mask 都在 _attn 裡面處理）
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        # 合回單頭 → output projection
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

#GPT2WithRoPE2D
#GPT2With2DSinusoids
class GPT2With2DSinusoids(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 用 Rope 版本替換每一層的 attention（並複製原層權重）
        for i, block in enumerate(self.transformer.h):
            rope_attn = RopeGPT2Attention(config)
            rope_attn.load_state_dict(block.attn.state_dict(), strict=False) #這樣它只會載入權重，忽略我們新加的 buffer：
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
        position_2d=None,           # [B, T, 2] (必需給定來開啟 RoPE2D)
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        # input_ids / inputs_embeds 二擇一
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.transformer.wte(input_ids)
            input_ids = None

        # 把 position_2d 掛到每層的 attn 上（讓 RopeGPT2Attention 讀取）
        if position_2d is not None:
            for block in self.transformer.h:
                block.attn._rope_position_2d = position_2d
        else:
            for block in self.transformer.h:
                block.attn._rope_position_2d = None

        # 呼叫原生 GPT2Model（會走我們替換後的 attn）
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    @classmethod
    def from_config(cls, config, **kwargs):
        """Create model from config with random weights"""
        return cls(config, **kwargs)