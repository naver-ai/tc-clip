"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/utils/optimizer.py
"""

import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    has_decay_names = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            has_decay.append(param)
            has_decay_names.append(name)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}], has_decay_names, no_decay_names


def filter_keywords(model, weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    params, names = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        params.append(param)
        names.append(name)

    return [{'params': params, 'weight_decay': weight_decay, 'lr': lr}], names


def build_optimizer(logger, config, model):
    model = model.module if hasattr(model, 'module') else model

    if not config.lr_10x:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay,
                                betas=tuple(config.betas), eps=1e-8, )
        return optimizer

    randomly_initialized_part = ["prompt_learner", "VPT", "prompt_generation"]

    param_groups = []
    params, name = filter_keywords(model, weight_decay=config.weight_decay, lr=config.lr,
                                   not_have=tuple(randomly_initialized_part))
    param_groups.extend(params)

    for part in randomly_initialized_part:
        params, name = filter_keywords(model, weight_decay=config.weight_decay,
                                       lr=config.lr * 10.,
                                       have=(part,))
        if len(name) == 0:
            continue
        param_groups.extend(params)
        logger.info(f"Set initial lr to 10x of other parts")
        logger.info(f"Initialize {config.lr * 10.:.2e} for params with keyword {part}")
        logger.info("Added:")
        for i in name:
            logger.info(i)
    optimizer = optim.AdamW(param_groups, betas=tuple(config.betas), eps=1e-8, )

    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)

    if config.lr_scheduler == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.lr_min,
            warmup_lr_init=0,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
