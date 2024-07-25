# Configuration
This project utilizes Hydra for setting up configurations. 
The `configs/` folder is organized as follows:
```
configs/
|–– aug/             # augmentation
|–– common/          # specific path to dataset, labels, etc.
|–– data/            # dataset protocol
|   |–– base2novel_hmdb51.yaml
|   |–– few_shot_ssv2.yaml
|   |–– fully_supervised_k400.yaml
|   |–– zero_shot_k400.yaml
|   └–– ...
|–– hydra_configs/   # hydra settings
|–– logging/         # logging
|–– optimizer/       # optimizer, scheduler
|–– trainer/         # model
|   |–– tc_clip.yaml
|   |–– vifi_clip.yaml
|   └–– ...
|
|–– base2novel.yaml
|–– few_shot.yaml
|–– fully_supervised.yaml
└–– zero_shot.yaml
```

- The yaml files under the `configs/` folder, i.e., {`base2novel.yaml`, `few_shot.yaml`, `fully_supervised.yaml`, `zero_shot.yaml`} are called **basic configs**,
where each basic config corresponds to a different protocol.
- The folders under the `configs/` folder, e.g., `aug/`, `common/`, contrains multiple configurations in each **default config** field, 
where the list of defaults are specified in each basic config as follows:

```yaml
defaults:
  - aug: default_aug  # augmentation
  - common: default   # common fields
  - data: zero_shot_k400  # data protocol
  - hydra_configs: default_hydra
  - logging: wandb    # logging
  - optimizer: adamw  # optimizer, scheduler
  - trainer: tc_clip  # model
  - _self_
...
```
- We start by selecting one of the basic configuration files by calling `-cn ${basic_config_name}` option
and then override default configurations by specifying the desired values by using `${default_config_field}=${default_config_name}`.
- For example:
```bash
torchrun --nproc_per_node=4 main.py -cn fully_supervised \
data=fully_supervised_k400 output=${your_ckpt_saving_path} trainer=tc_clip
```
- In this example, the `fully_supervised` basic config is chosen.
The default data config is overridden with `fully_supervised_k400`, 
the defualt trainer config is overriden with `tc_clip`,
and the output path variable is specified as `${your_ckpt_saving_path}`.
- By following this approach, you can easily configure the settings for your specific use case 
by selecting the appropriate basic config and overriding default configurations as needed.
- For more examples on running commands, refer to the training/evaluation sections of [README.md](README.md).


## Data Configuration
Under the `configs/data/` folder, there are data configuration YAML files named with `${protocol}_${dataset_name}.yaml`,
each corresponding to specific protocols and datasets.
Specifically, this `data` configuration is formatted like:
```yaml
data:
  train:
    dataset_name: k400_train
    root: ${k400.root}
    num_classes: ${k400.num_classes.default}
    label_file: ${k400.label_files.default}
    ann_file: ${k400.ann_files.train_replacement}
  val:
    dataset_name: k400_val
    ...
  test:
    - name: hmdb51_val
      protocol: avg_std
      dataset_list:
      - dataset_name: hmdb51_val1
        ...
      - dataset_name: hmdb51_val2
        ...
      - dataset_name: hmdb51_val3
        ...
    - name: ucf101_val
    ...
```
**Note**:
- During training, we build the train dataloader and validation dataloader using `${data.train}` and `${data.val}`. 
- If `main_testing` function is called, we iterate through multiple datasets in `${data.test}` for evaluation.
- The specific paths to each dataset, label files, and ann files in the `data` config are overriden from the `common` config. Thus, **after downloading the dataset, you must fill the values in `common` configs to your specific paths.**


## Logging Configuration
- This code supports W&B logging. To use W&B logging during training, fill out `${wandb_api_key}` with yours in `configs/logging/wandb.yaml` file.
- The running status will be logged under the `${wandb_project}`. The name of the run is same with the last folder name of `${output}`.
For example, if you run `torchrun --nproc_per_node=4 main.py output=workspace/expr/zero_shot_k400_tc_clip`, the run name is `zero_shot_k400_tc_clip`.