"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/clip/model.py
"""

import torch
from torch import nn

from clip.model_utils import LayerNorm
from clip.transformer import Transformer


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5

        # Prompt tokens at first layer
        # If vision_depth > 1, additional prompt tokens will be attached inside each block
        self.VPT_shallow = False if design_details["vision_depth"] == 0 else True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width) # [n_ctx, 768]
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            self.VPT.half()

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        pos_embedding_length = (input_resolution // patch_size) ** 2 + 1

        self.positional_embedding = nn.Parameter(scale * torch.randn(pos_embedding_length, width))  # e.g., [197, 768]
        self.ln_pre = LayerNorm(width)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_attention=False):
        # patch encoding
        B, T, C, H, W = x.size()
        x = x.reshape(-1, C, H, W)
        x = self.conv1(x)  # shape = [b*t, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Concat cls embedding
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Concat prompt tokens
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1)  # [128, 8, 768]
            x = torch.cat([x, visual_ctx], dim=1)   # -> [128, 205, 768] {CLS, patches, prompts}
        else:
            assert self.prompt_till_layer_visual == 0

        # Transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        attns = None
        if return_attention:
            x, attns = self.transformer.forward_return_attention(x)
        else:
            x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD     [BT, 197, D]
        x = self.ln_post(x[:, 0, :])    # [BT, 768]    Use cls token

        if self.proj is not None:
            x = x @ self.proj

        x = x.reshape(B, T, -1)

        return x, attns
