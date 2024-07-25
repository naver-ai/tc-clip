"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/trainers/vificlip.py
"""

import torch

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.print_utils import colorstr

from trainers.vifi_clip import ViFiCLIP
from trainers.tc_clip import TCCLIP


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.model_arch
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"vision_model": cfg.get("vision_model", "VisionTransformer"),
                      "vision_block": cfg.get("vision_block", "ResidualAttentionBlock"),
                      "text_block": cfg.get("text_block", "ResidualAttentionBlock"),
                      "use_custom_attention": cfg.get("use_custom_attention", False),
                      "context_length": cfg.get("context_length", 77),
                      "temporal_length": cfg.get("num_frames", 16),
                      "vision_depth": cfg.get("prompt_depth_vision", 0),
                      "language_depth": cfg.get("prompt_depth_text", 1),
                      "vision_ctx": cfg.get('n_ctx_vision', 0),
                      "language_ctx": cfg.get('n_ctx_text', 0),
                      # TC-CLIP
                      "positional_embedding_type": cfg.get("positional_embedding_type", "space"),
                      "local_global_bias": cfg.get("local_global_bias", True),
                      "context_token_k": cfg.get("context_token_k", 96),
                      "seed_token_a": cfg.get("seed_token_a", 0.3),
                      "tome_r": cfg.get("tome_r", 100),
                      "tome_d": cfg.get("tome_d", 0)
                      }

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def returnCLIP(config, logger=None, class_names=None, return_clip_model=False):
    logger.info(f"Loading CLIP (backbone: {config.model_arch})")
    clip_model = load_clip_to_cpu(config)

    logger.info(colorstr(f"Building {config.trainer_name}"))
    if config.trainer_name == "ViFiCLIP":
        model = ViFiCLIP(config, class_names, clip_model, logger)
    elif config.trainer_name == "TCCLIP":
        model = TCCLIP(config, class_names, clip_model, logger)
    else:
        raise NotImplementedError

    # Freeze parameters
    freeze_type = config.get("freeze", None)
    if freeze_type is not None:
        keyword_exception = ["prompt_learner", "VPT", 'prompt_generation']
        if freeze_type == 'backbone':
            logger.info(f"Turning off gradients for both encoders")
            module_to_freeze = ['image_encoder', 'text_encoder']
            module_to_update = ['prompt_learner', 'logit_scale']
        elif freeze_type == 'image':
            logger.info(f"Turning off gradients for image encoders")
            module_to_freeze = ['image_encoder']
            module_to_update = ['text_encoder', 'prompt_learner', 'logit_scale']
        elif freeze_type == 'text':
            logger.info(f"Turning off gradients for text encoders")
            module_to_freeze = ['text_encoder']
            module_to_update = ['image_encoder', 'prompt_learner', 'logit_scale']
        else:
            raise NotImplementedError

        for name, param in model.named_parameters():
            if check_keywords_in_name(name, module_to_update):  # part to update
                param.requires_grad_(True)
            elif check_keywords_in_name(name, module_to_freeze):    # part to freeze
                if check_keywords_in_name(name, keyword_exception):  # update several parts in freezed modules
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                raise NotImplementedError

    logger.info('----------------------------------------------------')
    logger.info('Freezed Parameters')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            logger.info(name)
    logger.info('----------------------------------------------------')

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Number of Parameters: {total_params / 1e6:.1f}M')

    model.float()

    if return_clip_model:
        return model, clip_model
    else:
        del clip_model
        torch.cuda.empty_cache()
        return model


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
