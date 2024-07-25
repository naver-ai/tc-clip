"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from einops import rearrange
import torch
from torch import nn

from clip.model_utils import LayerNorm
from clip.transformer import Transformer

from tome.utils import parse_r


class TCVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, num_frames: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.pos_emb_type = design_details["positional_embedding_type"]
        if self.pos_emb_type == "space":
            print("Using spatial positional embedding")
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        elif self.pos_emb_type == "joint":
            print("Using joint spatio-temporal positional embedding")
            self.positional_embedding = nn.Parameter(scale * torch.randn(num_frames,
                                                                         (input_resolution // patch_size) ** 2 + 1, width))
        else:
            raise NotImplementedError

        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.num_layers = layers
        self.tome_r = design_details["tome_r"]
        self.tome_d = design_details["tome_d"]
        self._tome_info = {
            "r": self.tome_r,
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": False,
            "class_token": False,
            "distill_token": False,
        }

    def add_positional_embedding(self, x, B, T):
        if self.pos_emb_type == "space":
            # add same positional encoding for different timestep
            x = x + self.positional_embedding.to(x.dtype)
        elif self.pos_emb_type == "joint":
            # add individual learnable positional encoding
            BT, N, width = x.size()
            x = x.reshape(B, T, N, width)
            x = x + self.positional_embedding.to(x.dtype)
            x = x.reshape(BT, N, width)
        else:
            raise NotImplementedError
        return x

    def forward(self, x: torch.Tensor, return_layer_num=None, return_attention=False, return_source=False):
        # patch encoding
        B, T, C, H, W = x.size()
        x = x.reshape(-1, C, H, W)
        x = self.conv1(x)  # shape = [b*t, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Concat cls embedding
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)

        # add positional embedding
        x = self.add_positional_embedding(x, B, T)
        x = rearrange(x, '(B T) N D -> B (T N) D', B=B, T=T)

        x = self.ln_pre(x)
        self._tome_info["r"] = parse_r(self.num_layers, (self.tome_r, self.tome_d))  # r scheduler
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        self._tome_info["trace_source"] = return_source
        x, attns, source = self.transformer.forward_tc(x,
                                                       tome_info=self._tome_info,
                                                       layer_num_list=return_layer_num,
                                                       return_attention=return_attention,
                                                       return_source=return_source)  # [n_layer, B, n+k, 768]

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        cls_tokens = x[:, :, :T, :]
        context_tokens = x[:, :, T:, :]
        return cls_tokens, context_tokens, attns, source
