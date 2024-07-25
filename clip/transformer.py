"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from einops import rearrange
import torch
from torch import nn

from clip.transformer_blocks_vifi import ResidualAttentionBlock
from clip.transformer_blocks_tc import TCAttentionBlock


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.T = design_details['temporal_length']

        # if vision layer
        if not text_layer:
            if design_details['vision_block'] == 'ResidualAttentionBlock':
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, True,
                                                                                  text_layer, i,
                                                                                  design_details) if prompts_needed > i
                                                 else ResidualAttentionBlock(width, heads, attn_mask, False,
                                                                                       text_layer, i, design_details)
                                                 for i in range(layers)])

            elif design_details['vision_block'] == 'TCAttentionBlock':
                self.resblocks = nn.Sequential(*[TCAttentionBlock(width, heads, attn_mask, i, design_details)
                                                 for i in range(layers)])

            else:
                raise NotImplementedError

        # if text layer
        else:
            if design_details['text_block'] == 'ResidualAttentionBlock':
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, True,
                                                                                  text_layer, i,
                                                                                  design_details) if prompts_needed > i
                                                 else ResidualAttentionBlock(width, heads, attn_mask, False,
                                                                                       text_layer, i, design_details)
                                                 for i in range(layers)])
            else:
                raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

    def forward_return_attention(self, x):
        attns_all = []
        for i, block in enumerate(self.resblocks):
            x, attns = block(x, return_attention=True)
            attns_all.append(attns)
        attns_all = torch.stack(attns_all, dim=0)
        return x, attns_all

    def forward_tc(self, x, tome_info: dict, layer_num_list: list,
                   return_attention=False, return_source=False):
        feats, attns, source = [], [], []
        T, N = self.T, x.size(1) // self.T
        for i, block in enumerate(self.resblocks):
            x, attn = block(x, tome_info=tome_info, return_attention=return_attention)
            if i in layer_num_list:
                feat, context_tokens = x[:, :T*N, :], x[:, T*N:, :]
                feat = rearrange(feat, 'B (T N) D -> B T N D', T=T, N=N)
                cls_tokens = feat[:, :, 0, :]
                feat = torch.cat([cls_tokens, context_tokens], dim=1)  # [B, T+n_context_tokens, D]
                feats.append(feat)
            if return_source:
                source.append(tome_info["source"].clone())
            if return_attention:
                attns.append(attn)
        feats = torch.stack(feats, dim=1)   # [B, n_layer, T+n_context_tokens, D]
        if return_attention:
            attns = torch.stack(attns, dim=1)
        return feats, attns, source
