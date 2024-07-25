"""
https://github.com/muzairkhattak/ViFi-CLIP/blob/main/datasets/build.py
"""

from functools import partial
from collections.abc import Mapping
from omegaconf import ListConfig

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import warnings
warnings.filterwarnings(action='ignore', module='mmcv', category=UserWarning)
from mmcv.parallel import collate

from .base_dataset import *

# PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1):
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_train_dataloader(logger, config):
    logger.info("Building train dataloader")
    target_data_config = config.data.train
    scale_resize = int(256 / 224 * config.input_size)
    flip = False if 'ssv2' in target_data_config.dataset_name else True

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.num_frames),   # TSN sampling strategy https://github.com/open-mmlab/mmaction2/issues/1379#issuecomment-1010938270
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.input_size,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.input_size, config.input_size), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5) if flip else None,
        dict(type='ColorJitter', p=config.aug.color_jitter),
        dict(type='GrayScale', p=config.aug.gray_scale),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
    train_pipeline = [p for p in train_pipeline if p is not None]

    train_data = VideoDataset(dataset_name=target_data_config.dataset_name,
                              ann_file=target_data_config.ann_file, data_prefix=target_data_config.root,
                              labels_file=target_data_config.label_file, pipeline=train_pipeline)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    # Set worker seed
    if config.get('worker_init_fn', False):
        init_fn = partial(
            worker_init_fn, num_workers=config.num_workers, rank=config.rank,
            seed=config.seed) if config.seed is not None else None
    else:
        init_fn = None

    # Due to some errors with using multi-workers in hmdb51, force num_workers=0 and pin_memory=false
    if ((config.selected_option.data in ["few_shot_hmdb51", "few_shot_hmdb51_llm"] and config.shot == 16) or
            (config.selected_option.data in ["base2novel_hmdb51", "base2novel_hmdb51_llm"] and config.base in [2, 3])):
        num_workers, pin_memory = 0, False
        init_fn = None
    else:
        num_workers, pin_memory = config.num_workers, config.pin_memory

    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.batch_size),
        worker_init_fn=init_fn
    )

    class_names = [class_name for i, class_name in train_data.classes]

    return train_data, train_loader, class_names


def build_val_dataloader(logger, config, target_data_config):
    logger.info(f"Building val dataloader")
    scale_resize = int(256 / 224 * config.input_size)
    collect_keys = ['imgs', 'label']
    if config.get('gather_filename', False):
        collect_keys.append('file_id')

    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.num_frames, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.input_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=collect_keys, meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.num_crop == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.input_size))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.input_size)
    if config.num_clip > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.num_frames,
                               multiview=config.num_clip)  # NUM_CLIP temporal views
    val_pipeline = [p for p in val_pipeline if p is not None]

    val_data = VideoDataset(dataset_name=target_data_config.dataset_name,
                            ann_file=target_data_config.ann_file,
                            data_prefix=target_data_config.root,
                            labels_file=target_data_config.label_file,
                            pipeline=val_pipeline,
                            return_filename=config.get('gather_filename', False))

    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    # Set worker seed
    if config.get('worker_init_fn', False):
        init_fn = partial(
            worker_init_fn, num_workers=config.num_workers, rank=config.rank,
            seed=config.seed) if config.seed is not None else None
    else:
        init_fn = None

    if 'hmdb51' in target_data_config.dataset_name or 'ucf101' in target_data_config.dataset_name:
        num_workers, pin_memory = 0, False
        init_fn = None
    else:
        num_workers, pin_memory = config.num_workers, config.pin_memory

    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=config.test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.test_batch_size),
        worker_init_fn=init_fn
    )

    class_names = [class_name for i, class_name in val_data.classes]
    return val_data, val_loader, class_names
