"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from collections import OrderedDict
from einops import rearrange
from timm.models.layers import trunc_normal_

import torch
from torch import nn

from clip.model_utils import LayerNorm, QuickGELU
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import schedule_r_constant


class TCAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, i=0, design_details=None):
        super().__init__()
        self.T = design_details['temporal_length']
        self.num_patches = 196
        self.num_context_token = design_details['context_token_k']  # total number of context token
        self.first_layer = (i == 0)
        self.attn = TCAttention(d_model, n_head, first_layer=self.first_layer,
                                T=self.T, seed_token_a=design_details["seed_token_a"],
                                local_global_bias=design_details['local_global_bias'])
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x, tome_info, return_attention=False):
        B, T, D = x.size(0), self.T, x.size(2)
        N = 1 + self.num_patches
        TN = T*N

        # TC-MHSA with seed index selection
        x_attn, seed_index, attn = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x[:, :TN, :] + x_attn  # B, TN, D

        # Summarize context tokens
        context_tokens = self.summarize_context_tokens(x, seed_index, tome_info=tome_info)    # [B, num_context_tokens, D]

        # FFN
        x = torch.cat([x, context_tokens], dim=1)
        x = x + self.mlp(self.ln_2(x))
        return x, attn

    def summarize_context_tokens(self, x, index, tome_info):
        # x: B, TN, D
        # index: B, T, m
        B, T, N, D = x.size(0), self.T, x.size(1) // self.T, x.size(-1)
        patch_tokens = rearrange(x, 'B (T N) D -> B T N D', T=T, N=N)
        patch_tokens = patch_tokens[:, :, 1:1+self.num_patches, :]     # x: {cls(1), patch_tokens(196)}
        Np = patch_tokens.size(2)  # number of patch tokens in each frame (196)
        Ns = index.size(2)         # number of seed tokens in each frame

        # Gather seed tokens from all frames
        index_expand = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(2).repeat(B, 1, Ns)*Np
        index_flatten = (index + index_expand).reshape(B, -1)
        patch_tokens = rearrange(patch_tokens, 'B T Np D -> B (T Np) D')
        seed_tokens = patch_tokens.gather(dim=1, index=index_flatten.unsqueeze(-1).expand(-1, -1, D))    # [B, T*Ns, D]

        # Bipartite matching with seed tokens
        default_r = tome_info["r"].pop(0)
        r_schedule, cnt_schedule = schedule_r_constant(start_number=Ns * T, final_number=self.num_context_token, r=default_r)
        tome_info['source'] = None
        tome_info['size'] = None
        for i, r in enumerate(r_schedule):
            metric = seed_tokens.clone()
            merge, _ = bipartite_soft_matching(metric, r, tome_info["class_token"], tome_info["distill_token"])
            if tome_info['trace_source']:
                tome_info['source'] = merge_source(merge, seed_tokens, tome_info['source'])
            seed_tokens, tome_info['size'] = merge_wavg(merge, seed_tokens, tome_info["size"])
        context_tokens = seed_tokens

        # Convert source matrix to original T*num_patches
        if tome_info['trace_source']:
            source_all = torch.zeros(B, context_tokens.size(1), T * Np, device=x.device)
            source_all.scatter_(dim=2, index=index_flatten.unsqueeze(1).expand(-1, seed_tokens.size(1), -1), src=tome_info['source'])
            tome_info['source'] = source_all
        return context_tokens


class TCAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, T=16, seed_token_a=0.3, first_layer=False, local_global_bias=False):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = n_head
        self.head_dim = d_model // n_head

        self.T = T
        self.num_patches = 196
        self.top_s = int(self.num_patches * seed_token_a)    # number of seed tokens in each frame
        self.first_layer = first_layer

        if not self.first_layer and local_global_bias:    # [num_heads, 2(local, global)]
            self.local_global_bias_table = nn.Parameter(torch.zeros(n_head, 1, 2))
            trunc_normal_(self.local_global_bias_table, std=.02)
        else:
            self.local_global_bias_table = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def get_seed_index(self, attn):
        B, T = attn.size(0) // self.T, self.T
        cls_attn = attn[:, :, 0, 1:1 + self.num_patches]  # [BT, head, num_patches]
        cls_attn = cls_attn.reshape(B, T, self.num_heads, self.num_patches)
        cls_attn = cls_attn.mean(dim=2)     # [B, T, num_patches]
        _, idx = torch.topk(cls_attn, self.top_s, dim=2, largest=True, sorted=True)  # [B, T, top_s]
        return idx

    def forward(self, x: torch.Tensor, return_attention=False):
        # x: [B, T*N, D] or [B, T*N+k, D]
        B, L, C = x.shape
        T, N = self.T, 1+self.num_patches
        BT, TN = B*T, T*N

        # in-projection
        q = self.q_proj(x[:, :TN, :])  # [B, T*N, D]
        k = self.k_proj(x)  # [B, T*N, D] or [B, T*N+k, D]
        v = self.v_proj(x)  # [B, T*N, D] or [B, T*N+k, D]

        # Repeat context tokens for temporal axis
        q = q.reshape(BT, N, C)
        if self.first_layer:
            k = k.reshape(BT, N, C)
            v = v.reshape(BT, N, C)
        else:
            k_local = k[:, :TN, :].reshape(BT, N, C)
            v_local = v[:, :TN, :].reshape(BT, N, C)
            k_context = k[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)    # [B, k, D] -> [BT, k, D]
            v_context = v[:, TN:, :].unsqueeze(1).repeat(1, T, 1, 1).reshape(BT, -1, C)
            k = torch.cat([k_local, k_context], dim=1)   # [BT, N+k, D]
            v = torch.cat([v_local, v_context], dim=1)

        q = q.reshape(BT, q.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N, C // num_heads]
        k = k.reshape(BT, k.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N+k, C // num_heads]
        v = v.reshape(BT, v.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # [B, num_heads, N+k, C // num_heads]

        # attention
        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)    # [BT, nhead, Nq, Nk]

        # Add Local-global bias
        if self.local_global_bias_table is not None:
            # expand [num_heads, 1, 2] -> [bt, nhead, Nq, Nk]
            local_bias = self.local_global_bias_table[:, :, 0:1].unsqueeze(0).repeat(BT, 1, attn.size(2), attn.size(2))
            global_bias = self.local_global_bias_table[:, :, 1:].unsqueeze(0).repeat(BT, 1, attn.size(2), attn.size(3) - attn.size(2))
            bias = torch.cat([local_bias, global_bias], dim=-1)
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(BT, q.size(2), C)

        # out-projection
        x = x.reshape(B, TN, C)
        x = self.out_proj(x)

        # Select top-s indices with CLS attention score
        index = self.get_seed_index(attn)

        if return_attention:
            return x, index, attn[:, :, :, :N]
        else:
            return x, index, None
