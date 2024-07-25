"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/trainers/vificlip.py
"""

import re
import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class ViFiCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32

    def _rebuild_classnames(self, cfg, classnames, clip_model, logger):
        self.prompt_learner._rebuild_classnames(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    def forward(self, image, return_attention=False):
        tokenized_prompts = self.tokenized_prompts  # (num_classes, token_len)
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner() # (num_classes, token_len, channel) ex. (51, 77, 512)

        # Encode image features
        image_features, attns = self.image_encoder(image.type(self.dtype),
                                                   return_attention=return_attention)

        # Now take the mean along the temporal direction
        image_features = image_features.mean(dim=1, keepdim=False)

        # Encode text features
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # Calculate logits
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [b, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [num_class, 512]
        logits = logit_scale * image_features @ text_features.t()   # [b, num_class]

        return {"logits": logits,
                "attention": attns}


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype if cfg.opt_level != 'O0' else torch.float32
        self.context_length = cfg.get('context_length', 77)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding[:self.context_length, ...].type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.prompt_model
        ctx_init = cfg.ctx_init
        if cfg.get('parse_ucf101', True):  # Use regular expression to insert space before each capital letter
            classnames = [re.sub(r'([A-Z])', r' \1', name).strip() for name in classnames]
        ZS_evaluation = cfg.zs_eval
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=cfg.get("context_length", 77))
                                           for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.prompt_depth_text >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = cfg.n_ctx_text
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init, context_length=cfg.get("context_length", 77))
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.prompt_prefix = " ".join(["X"] * n_ctx)
            logger.info(f"V-L design")
            logger.info(f'Initial text context: "{self.prompt_prefix}"')
            logger.info(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            logger.info(f"Number of context words (tokens) for Vision prompting: {cfg.n_ctx_vision}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            # No prompting
            ctx_init = ctx_init.replace("_", " ")
            self.prompt_prefix = ctx_init
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def _rebuild_classnames(self, cfg, classnames, clip_model, logger):
        # During zero-shot evaluation, just replace classnames instead of building whole model
        logger.info(f"Rebuild {len(classnames)} classnames")
        self.use_prompt_stage = cfg.prompt_model
        if cfg.get('parse_ucf101', True):  # Use regular expression to insert space before each capital letter
            classnames = [re.sub(r'([A-Z])', r' \1', name).strip() for name in classnames]
        ZS_evaluation = cfg.zs_eval
        if ZS_evaluation:
            # Rebuild complete text embedding without context vectors
            dtype = self.complete_text_embeddings.dtype
            device = self.complete_text_embeddings.device
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=cfg.get("context_length", 77))
                                           for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            dtype = self.token_prefix.dtype
            device = self.token_prefix.device
            # Rebuild token_prefix, token_suffix
            n_cls = len(classnames)
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)

            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        else:
            dtype = self.complete_text_embeddings.dtype
            device = self.complete_text_embeddings.device
            # No prompting. Rebuild complete text embeddings
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).to(device)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor

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
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts
