"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import re
import copy
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class VPPromptLearner(nn.Module): 
    """reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/trainers/vificlip.py"""
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        n_cls = len(classnames)
        self.n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if cfg.get('parse_ucf101', True):  # Use regular expression to insert space before each capital letter
            classnames = [re.sub(r'([A-Z])', r' \1', name).strip() for name in classnames]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            ctx_init_len = len(ctx_init.split())
            prompt = clip.tokenize(ctx_init, context_length=cfg.get("context_length", 77))
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + ctx_init_len, :]
            self.prompt_prefix = ctx_init
            if ctx_init_len < self.n_ctx:
                # add random initialization vectors
                ctx_vectors_extra = torch.empty(self.n_ctx - ctx_init_len, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors_extra, std=0.02)
                ctx_vectors = torch.cat([ctx_vectors, ctx_vectors_extra], dim=0)
                self.prompt_prefix = ctx_init + " " + " ".join(["X"] * (self.n_ctx - ctx_init_len))
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        logger.info('Video-conditional prompt learning')
        logger.info(f'Initial context: "{self.prompt_prefix}"')
        logger.info(f"Number of learnable text prompt vectors: {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def _rebuild_classnames(self, cfg, classnames, clip_model, logger):
        # During zero-shot evaluation, just replace classnames instead of building whole model
        logger.info(f"Rebuild {len(classnames)} classnames")
        dtype = self.token_prefix.dtype
        device = self.token_prefix.device
        if cfg.get('parse_ucf101', True):  # Use regular expression to insert space before each capital letter
            classnames = [re.sub(r'([A-Z])', r' \1', name).strip() for name in classnames]

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    """reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/trainers/vificlip.py"""    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts

