"""
reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/utils/tools.py
"""

import copy
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import clip
import os


def init_dist():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl')
    dist.barrier()


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_main():
    rank, _ = get_dist_info()
    return rank == 0


def set_random_seed(seed, use_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Setting random seed: {}".format(seed))
    if use_cudnn:
        if is_main():
            print("Using CuDNN Benchmark")
        cudnn.benchmark = True
        cudnn.enabled = True
    else:
        if is_main():
            print("Disabling CuDNN Benchmark")
        cudnn.deterministic = True
        cudnn.benchmark = False


def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def accuracy_top1_top5(pred, target):
    b = pred.size(0)
    values_1, indices_1 = pred.topk(1, dim=-1)
    values_5, indices_5 = pred.topk(5, dim=-1)
    acc1, acc5, count = 0, 0, 0
    for i in range(b):
        gt = target[i]
        if not isinstance(gt, list):
            if gt < 0:
                continue
            gt = [gt]
        if len(gt) == 0:
            continue
        count += 1
        if indices_1[i] in gt:
            acc1 += 1
        if any(idx.item() in gt for idx in indices_5[i]):
            acc5 += 1
    return acc1, acc5, indices_1, count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    # epoch saving
    if config.save_intermediate and (epoch + 1) % 10 == 0 and (epoch + 1) != config.epochs:
        save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch+1}.pth')
        logger.info(f"{save_path} saving......")
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved !!!")

    save_path = os.path.join(working_dir, f'last.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")

        
def wise_state_dict(logger, ori_model, loaded_state_dict, weight_for_ft=None, keywords_to_exclude=None):
    """reference: https://github.com/mlfoundations/wise-ft"""
    if keywords_to_exclude is None:
        keywords_to_exclude = []

    finetuned_model = copy.deepcopy(ori_model)
    msg = finetuned_model.load_state_dict(loaded_state_dict, strict=False)
    logger.info(f'load finetuned model {msg}')

    state_dict_ori = dict(ori_model.named_parameters())
    state_dict_finetuned = dict(finetuned_model.named_parameters())
    assert set(state_dict_ori) == set(state_dict_finetuned)

    fused_dict = {}
    for k in state_dict_ori:
        # Check if the current parameter name contains any of the excluded keywords
        if any(keyword in k for keyword in keywords_to_exclude):
            # If it does, use the finetuned model's parameters directly without fusion
            logger.info(f'weight fusion exception: {k}')
            fused_dict[k] = state_dict_finetuned[k]
        else:
            # Otherwise, fuse the weights according to the specified ratio
            fused_dict[k] = (1 - weight_for_ft) * state_dict_ori[k] + weight_for_ft * state_dict_finetuned[k]
    return fused_dict


def load_checkpoint(config, model, optimizer, lr_scheduler, logger, model_only=False):
    if os.path.isfile(config.resume):
        logger.info(f"==============> Resuming from {config.resume}....................")
        checkpoint = torch.load(config.resume, map_location='cpu')
        load_state_dict = checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in load_state_dict:
            del load_state_dict["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
            del load_state_dict["module.prompt_learner.complete_text_embeddings"]

        # Remove "module." prefix if present
        if not hasattr(model, 'module'):
            load_state_dict = {k.replace('module.', ''): v for k, v in load_state_dict.items()}

        """reference: https://github.com/mlfoundations/wise-ft"""
        if config.get('wise_ft', 0.0) != 0:
            keywords_to_exclude = ['local_global_bias', 'prompt_learner', 'prompt_generation']
            fused_state_dict = wise_state_dict(logger, ori_model=model, loaded_state_dict=load_state_dict,
                                               weight_for_ft=config.get('wise_ft', 0.0),
                                               keywords_to_exclude=keywords_to_exclude)
            msg = model.load_state_dict(fused_state_dict, strict=False)
            logger.info(f"Wise FT weight for fine-tuned model {config.get('wise_ft', 0.0)}, fused model {msg}")
        else:
            msg = model.load_state_dict(load_state_dict, strict=False)
            logger.info(f"resume model: {msg}")

        try:
            if not model_only:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.resume}' (epoch {checkpoint['epoch']})")

            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.resume)))
        raise Exception


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file
