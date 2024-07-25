"""
TC-CLIP
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from datasets.build import build_train_dataloader, build_val_dataloader
from datasets.blending import CutmixMixupBlending
from trainers.build_trainer import returnCLIP
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import epoch_saving, load_checkpoint, is_main, init_dist, get_dist_info, set_random_seed
from utils.logger import create_logger
from utils.print_utils import colorstr, print_configs

from engine import train_one_epoch, validate


def main_training(logger, config):
    "------------ Build dataloader, criterion -----------"
    train_data, train_loader, class_names = build_train_dataloader(logger, config)
    val_data, val_loader, _ = build_val_dataloader(logger, config, target_data_config=config.data.val)

    mixup_fn = None
    if config.aug.mixup > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.data.train.num_classes,
                                       smoothing=config.aug.label_smooth,
                                       mixup_alpha=config.aug.mixup,
                                       cutmix_alpha=config.aug.cutmix,
                                       switch_prob=config.aug.mixup_switch_prob)
    elif config.aug.label_smooth > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.aug.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()

    "------------ Build model, optimizer, scheduler -----------"
    model = returnCLIP(config, logger, class_names)
    model = model.cuda()
    optimizer = build_optimizer(logger, config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.opt_level != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.opt_level)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank], broadcast_buffers=False,
                                                      find_unused_parameters=False)

    "------------ Load checkpoint -----------"
    start_epoch, max_accuracy, max_accuracy_acc5, load_model_only = 0, 0.0, 0.0, True
    if config.auto_resume:
        resume_file = os.path.join(config.output, 'last.pth')   # resume from last.pth
        if resume_file:
            config.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
            load_model_only = False
        else:
            logger.info(f'no checkpoint found in {config.output}, ignoring auto resume')

    if config.resume:
        start_epoch, max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler,
                                                    logger, model_only=load_model_only)
        if not config.auto_resume and start_epoch > 1:    # reset epoch only when finetuning
            logger.info("resetting epochs no and max. accuracy to 0 after loading pre-trained weights")
            start_epoch = 0
            max_accuracy = 0

    "------------ Eval only mode -----------"
    if config.eval is not None:
        test_stats = validate(val_loader, model, logger, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: Acc@1 {test_stats['acc1']:.1f}, "
                    f"Acc@5 {test_stats['acc5']:.1f}\n")
        return

    "------------ Training mode -----------"
    for epoch in range(start_epoch, config.epochs):
        train_loader.sampler.set_epoch(epoch)

        # train
        train_stats = train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, logger, config, mixup_fn)
        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        logger.info("\n")
        if is_main() and config.use_wandb:
            wandb.log(log_stats, step=(epoch + 1) * len(train_loader) - 1)

        # validation
        if epoch % config.save_freq == 0 or epoch == (config.epochs - 1) or epoch == start_epoch:
            test_stats = validate(val_loader, model, logger, config)
            acc1, acc5 = test_stats['acc1'], test_stats['acc5']
            logger.info(f"Accuracy of the network on the {len(val_data)} test videos: Acc@1 {acc1:.1f}, "
                        f"Acc@5 {acc5:.1f}\n")

            is_best = acc1 > max_accuracy
            max_accuracy = acc1 if is_best else max_accuracy
            max_accuracy_acc5 = acc5 if is_best else max_accuracy_acc5
            logger.info(f'Max accuracy: {max_accuracy:.2f}%\n')
            if is_main() and (epoch % config.save_freq == 0 or epoch == (config.epochs - 1) or is_best):
                epoch_saving(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, config.output,
                             is_best)

            log_stats = {'val/acc1': acc1,
                         'val/acc5': acc5,
                         'val/best': max_accuracy,
                         'val/best5': max_accuracy_acc5}

            if is_main() and config.use_wandb:
                wandb.log(log_stats, step=(epoch + 1) * len(train_loader) - 1)

            # early stopping
            if config.early_stop and acc1 < max_accuracy:
                logger.info(f"Early stopping at epoch {epoch}...")
                break

    del model
    torch.cuda.empty_cache()

    "------------ Final testing with best checkpoint -----------"
    if config.final_test and 'test' in config.data:   # test best checkpoint
        config.resume = os.path.join(config.output, 'best.pth')
        main_testing(logger, config)

        # weight-space ensembling
        if config.protocol == 'zero_shot':
            config.wise_ft = 0.7
            main_testing(logger, config, prefix='test_w0.7')

    return


def main_testing(logger, config, prefix='test'):
    if config.protocol == 'fully_supervised' and config.multi_view_inference:
        config.num_clip = 4
        config.num_crop = 3
    elif config.protocol == 'zero_shot' and config.multi_view_inference:
        config.num_clip = 2

    if config.num_clip != 1 or config.num_crop != 1:
        logger.info(f"======== Testing with multi-view inference: "
                    f"{config.num_frames}x{config.num_clip}x{config.num_crop} ========")

    model, clip_model = None, None
    result_dict = {}
    total_acc1_list = []
    for dataset_config in config.data.test:
        name = dataset_config.name  # ex. hmdb51_val
        protocol = dataset_config.get("protocol", "top1")
        acc1_list, acc5_list = [], []
        for test_config in dataset_config.dataset_list:
            dataset_name = test_config.dataset_name

            logger.info(f"======== Start evaluation on {colorstr(dataset_name)} =======")

            "------------ Build dataloader, model -----------"
            val_data, val_loader, class_names = build_val_dataloader(logger, config, target_data_config=test_config)

            # At first iteration, build model & load checkpoints
            if model is None:
                model, clip_model = returnCLIP(config, logger, class_names, return_clip_model=True)
                model.cuda()

                if config.opt_level != 'O0':
                    model = amp.initialize(models=model, opt_level=config.opt_level)

                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank], broadcast_buffers=False,
                                                                  find_unused_parameters=False)

                if config.resume:
                    epoch_loaded, max_accuray_loaded = load_checkpoint(config, model, None, None, logger, model_only=True)
                    logger.info(
                        f"Loaded checkpoint at epoch {epoch_loaded} with max accuracy {max_accuray_loaded:.1f}")

            # From second iteration, just rebuild classnames part only
            else:
                model.module._rebuild_classnames(config, class_names, clip_model, logger)

            "------------ Validation -----------"
            test_stats = validate(val_loader, model, logger, config)
            acc1_list.append(test_stats['acc1'])
            acc5_list.append(test_stats['acc5'])
            logger.info(f"Accuracy of the checkpoint on {colorstr(dataset_name)} test videos (size: {len(val_data)}): "
                        f"Acc@1 {test_stats['acc1']:.1f}, Acc@5 {test_stats['acc5']:.1f}")
            del val_loader
            del val_data
            torch.cuda.empty_cache()

        if protocol == "avg_std":
            result_dict[name] = {'acc1_avg': np.mean(acc1_list), 'acc1_std': np.std(acc1_list),
                                 'acc5_avg': np.mean(acc5_list), 'acc5_std': np.std(acc5_list), 'protocol': protocol}
            total_acc1_list.append(np.mean(acc1_list))
        else:
            result_dict[name] = {'acc1': acc1_list[-1], 'acc5': acc5_list[-1], 'protocol': protocol}
            total_acc1_list.append(acc1_list[-1])

    "------------ Log results -----------"
    if is_main() and config.use_wandb:
        wandb.log({f'{prefix}/acc1_total': np.mean(total_acc1_list),
                   f'{prefix}/mean': (max_accuray_loaded + np.mean(total_acc1_list)) / 2.})
    for name, result in result_dict.items():
        protocol = result.pop('protocol')
        if protocol == "avg_std":
            logger.info(f"Accuracy of the checkpoint on {name} test videos: "
                        f"Acc@1 {result['acc1_avg']:.1f} (+- {result['acc1_std']:.1f}), Acc@5 {result['acc5_avg']:.1f} (+- {result['acc5_std']:.1f})\n")
        else:
            logger.info(f"Accuracy of the checkpoint on {name} test videos: "
                        f"Acc@1 {result['acc1']:.1f}, Acc@5 {result['acc5']:.1f}\n")

        if len(result_dict) > 1:
            log_stats = {f"{prefix}/{name}_{k}": v for k, v in result.items()}
        else:
            log_stats = {f"{prefix}/{k}": v for k, v in result.items()}
        if is_main() and config.use_wandb:
            wandb.log(log_stats)

    return


@hydra.main(version_base=None, config_path="configs", config_name="zero_shot")
def main(config: DictConfig) -> None:
    if config.eval is None and config.protocol in ['zero_shot', 'few_shot', 'base2novel', 'fully_supervised']:
        assert config.protocol in config.selected_option.data, "Selected data should be same with the protocol"
    if config.protocol == "few_shot":
        assert config.shot in [2, 4, 8, 16], "Number of shot 'config.shot' should be defined"
    if config.protocol == "base2novel":
        assert config.base in [1, 2, 3], "Base seed 'config.base' should be defined"

    OmegaConf.set_struct(config, False)  # Needed to add fields at runtime below

    # Force num_workers=4 in hmdb51
    if 'hmdb51' in config.selected_option.data:
        config.num_workers = 4

    # Init DDP
    if os.getenv('RANK') is None:
        raise Exception("This code only supports DDP mode. Try with DDP")
    init_dist()

    # Define working dir
    Path(config.output).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.output, dist_rank=dist.get_rank(), name=f"{config.trainer_name}")
    logger.info(f"working dir: {config.output}")

    config.rank, config.world_size = get_dist_info()
    config.num_gpus = config.world_size
    if config.num_gpus == 1:
        logger.info(colorstr('Single GPU'))
        config.distributed = False
    else:
        logger.info(colorstr('DDP')+f' with {config.num_gpus} GPUs')
        config.distributed = True

    # Random seed
    if config.seed is not None:
        set_random_seed(config.seed + config.rank, use_cudnn=config.use_cudnn)

    # Set accumulation steps
    config.accumulation_steps = config.total_batch_size // (config.num_gpus*config.batch_size)
    logger.info(f"Total batch size ({config.total_batch_size}) "
                f"= num_gpus ({config.num_gpus}) * batch_size ({config.batch_size}) "
                f"* accumulation_steps ({config.accumulation_steps})")

    # wandb logger
    if config.eval is not None or config.get('debug', False):
        config.use_wandb = False
    elif is_main() and config.use_wandb:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        expr_name = os.path.split(config.output)[-1]
        tags = [f"{config.shot}shot" if config.protocol == "few_shot" else None,
                f"s{config.base}" if config.protocol == "base2novel" else None]
        tags = [t for t in tags if t is not None]
        tags.extend(config.get('wandb_tags', []))
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(name=expr_name, project=config.wandb_project, dir=config.wandb_logging_dir,
                   config=cfg_dict, tags=tags)

    # print configs
    print_configs(logger, config)

    if config.eval is None:
        main_training(logger, config)
    elif config.eval == "val":
        main_training(logger, config)
    elif config.eval == "test":
        main_testing(logger, config)
    else:
        raise NotImplementedError

    if is_main():
        wandb.finish()


if __name__ == '__main__':
    main()
