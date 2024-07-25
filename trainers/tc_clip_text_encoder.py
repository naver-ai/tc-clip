"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import copy
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PromptGenerationLayer(nn.Module):
    def __init__(self, d_model: int, d_cross: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=d_cross, vdim=d_cross, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.ln_1_kv = LayerNorm(d_cross)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.apply(_init_weights)

    def cross_attention(self, x: torch.Tensor, x2: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x2, x2, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, x2: torch.Tensor):
        x = x + self.cross_attention(self.ln_1(x), self.ln_1_kv(x2))
        x = x + self.mlp(self.ln_2(x))
        return x


class VPTextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, logger):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32
        self.context_length = cfg.get('context_length', 77)

        "=========== vision-conditional prompt generation ============"
        self.n_ctx = cfg.n_ctx
        self.im_stop_grad = cfg.im_stop_grad
        if isinstance(cfg.prompt_generation_layer_level, str):
            self.prompt_generation_layer_level = list(map(int, cfg.prompt_generation_layer_level.split('+')))
        elif isinstance(cfg.prompt_generation_layer_level, int):
            self.prompt_generation_layer_level = [cfg.prompt_generation_layer_level]
        else:
            self.prompt_generation_layer_level = cfg.prompt_generation_layer_level
        self.prompt_generation_layer_level = [i-1 for i in self.prompt_generation_layer_level]

        single_layer = PromptGenerationLayer(d_model=512, d_cross=512, n_head=8,
                                             attn_mask=None,
                                             dropout=0.0)
        self.prompt_generation_layer = _get_clones(single_layer, len(self.prompt_generation_layer_level))

        self.prompt_generation_layer_init = cfg.get("prompt_generation_layer_init", "random")
        if self.prompt_generation_layer_init == "clip":
            for i, num_layer in enumerate(self.prompt_generation_layer_level):
                logger.info(f"Copy CLIP transformer {num_layer}th layer weights to prompt generation layer")
                state_dict = clip_model.transformer.resblocks[num_layer].state_dict()
                state_dict['ln_1_kv.weight'] = state_dict['ln_1.weight'].clone()
                state_dict['ln_1_kv.bias'] = state_dict['ln_1.bias'].clone()
                try:
                    self.prompt_generation_layer[i].load_state_dict(state_dict)
                except:
                    missing_keys, _ = self.prompt_generation_layer[i].load_state_dict(state_dict, strict=False)
                    logger.info(f'Layer {i}: Weights not found for some missing keys: ', missing_keys)

        logger.info(f"Prompt generation level: {self.prompt_generation_layer_level}")
        logger.info(f"Prompt generation stop grad: {self.im_stop_grad}")

    def forward(self, prompts, tokenized_prompts, im_features):
        # im_features: n_layer, k, d
        x = prompts + self.positional_embedding[:self.context_length, ...].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.im_stop_grad:
            im_features = im_features.clone().detach()

        count = 0   # prompt generation layer count
        for i, block in enumerate(self.transformer.resblocks):
            if i in self.prompt_generation_layer_level:
                # text prompt generation with video features
                prefix = x[:1, :, :]    # SOS
                ctx = x[1:1+self.n_ctx, :, :]   # prompts
                suffix = x[1+self.n_ctx:, :, :]  # CLS, EOS
                im_features_cur = im_features[count, :, :].unsqueeze(1).expand(-1, ctx.size(1), -1)  # -> [k, n_cls, 512]
                ctx = self.prompt_generation_layer[count](x=ctx, x2=im_features_cur)
                count += 1

                x = torch.cat([prefix, ctx, suffix], dim=0)

            x = block(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, context_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
