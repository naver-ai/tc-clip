"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/clip/model.py
"""

from collections import OrderedDict

import torch
from torch import nn

from clip.model_utils import LayerNorm, QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor, return_attention=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if return_attention:
            x, attns = self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)  # attns: [b, num_head, L(#_queries), S(#_values)]
        else:
            x, attns = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)  # attns: None
        return x, attns

    def forward(self, x: torch.Tensor, return_attention=False):
        if self.add_prompt:
            if not self.text_layer:
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        y, attns = self.attention(self.ln_1(x), return_attention=return_attention)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        if return_attention:
            return x, attns
        else:
            return x
