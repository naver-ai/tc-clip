"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/main.py
"""

import wandb
from apex import amp
import torch
import torch.distributed as dist

from utils.tools import accuracy_top1_top5
from utils.logger import MetricLogger, SmoothedValue


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, logger, config, mixup_fn):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = 'Epoch: [{}]'.format(epoch)

    for idx, batch_data in enumerate(metric_logger.log_every(train_loader, config.print_freq, logger, header)):
        images = batch_data['imgs'].cuda(non_blocking=True)
        label_id = batch_data['label'].cuda(non_blocking=True)
        label_id = label_id.reshape(-1) # [b]
        images = images.view((-1, config.num_frames, 3) + images.size()[-2:])  # [b, t, c, h, w]

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)   # label_id [b] -> [b, num_class]

        # forward
        output = model(images)
        total_loss = criterion(output["logits"], label_id)
        total_loss_divided = total_loss / config.accumulation_steps

        # backward
        if config.accumulation_steps == 1:
            optimizer.zero_grad()
        if config.opt_level != 'O0':
            with amp.scale_loss(total_loss_divided, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss_divided.backward()
        if config.accumulation_steps > 1:
            if (idx + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        log_stats = metric_logger.get_stats(prefix='train_inner/')
        if dist.get_rank() == 0 and config.use_wandb:
            wandb.log(log_stats, step=epoch*num_steps+idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.get_stats()


@torch.no_grad()
def validate(val_loader, model, logger, config):
    model.eval()
    num_classes = len(val_loader.dataset.classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'

    logger.info(f"{config.num_clip * config.num_crop} views inference")
    for idx, batch_data in enumerate(metric_logger.log_every(val_loader, config.print_freq, logger, header)):
        _image = batch_data["imgs"]  # [b, tn, c, h, w]
        label_id = batch_data["label"]
        label_id = label_id.reshape(-1)  # [b]

        b, tn, c, h, w = _image.size()
        t = config.num_frames
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, num_classes)).cuda()
        for i in range(n):
            image = _image[:, i, :, :, :, :]  # [b,t,c,h,w]
            label_id = label_id.cuda(non_blocking=True)
            image_input = image.cuda(non_blocking=True)

            if config.opt_level == 'O2':
                image_input = image_input.half()

            output = model(image_input)
            logits = output["logits"]
            similarity = logits.view(b, -1).softmax(dim=-1)
            tot_similarity += similarity

        # Classification score
        acc1, acc5, indices_1, _ = accuracy_top1_top5(tot_similarity, label_id)
        metric_logger.meters['acc1'].update(float(acc1) / b * 100, n=b)
        metric_logger.meters['acc5'].update(float(acc5) / b * 100, n=b)

    metric_logger.synchronize_between_processes()
    logger.info(f' * Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')
    return metric_logger.get_stats()
