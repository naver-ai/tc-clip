# Training and Evaluation

Make sure to configure the dataset paths in `TC-CLIP/configs/common/default.yaml` and run the commands from the main directory `TC-CLIP/`.
Below we provide training instructions for TC-CLIP and its variants.
All command examples for TC-CLIP are listed under `TC-CLIP/scripts/train` and `TC-CLIP/scripts/eval`.

Instructions:
- [Zero-Shot Protocol](#zero-shot-protocol)
- [Few-Shot Protocol](#few-shot-protocol)
- [Base-to-Novel Protocol](#base-to-novel-protocol)
- [Fully-Supervised Protocol](#fully-supervised-protocol)


## Zero-Shot Protocol

We train all models on Kinetics-400 for 10 epochs 
and then evaluate directly on downstream datasets (i.e., HMDB-51, UCF-101, and K-600).

### Training in Zero-Shot Protocol

To train the model under the zero-shot protocol, run the following command:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn zero_shot \
data=${protocol}_${dataset_name} output=${your_ckpt_saving_path} trainer=${trainer_name}

# Example:
torchrun --nproc_per_node=4 main.py -cn zero_shot \
data=zero_shot_k400 output=ckpt/zero_shot_k400_tc_clip trainer=tc_clip
```

Note: After training, the `main_testing` function is automatically called twice to evaluate the best checkpoint in the zero-shot evaluation protocol.
- During the first call, the evaluation is conducted with the fine-tuned model.
- During the second call, the evaluation is performed with the weight-space ensemble between CLIP and the fine-tuned TC-CLIP.
The weight ratio is controlled by the `wise_ft` variable, which is set to 0.7 by default.

### Evaluation in Zero-Shot Protocol

After training, you also have the option to manually evaluate with the saved checkpoint using the following command:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn zero_shot \
data=${protocol}_${dataset_name} output=${your_result_saving_path} trainer=${trainer_name} \
eval=test wise_ft=${wise_ft} resume=${ckpt_path}

# Example for wise-ft:
torchrun --nproc_per_node=4 main.py -cn zero_shot \
data=zero_shot_k400 output=/PATH/TO/OUTPUT trainer=tc_clip \
eval=test wise_ft=0.7 resume=ckpt/zero_shot_k400_tc_clip/best.pth
```
In this command:
- `eval=test` and `resume=${ckpt_path}` are added to run the evaluation-only mode with the best chekpoint. 
- If ${wise_ft} is set to 0.0, it means that no weight-space ensemble is applied.
- If ${wise_ft} is set to 1.0, it means that the vanilla CLIP weight is used.



## Few-Shot Protocol

In the few-shot protocol, models are directly trained and evaluated on downstream datasets, including HMDB-51, UCF-101, and SSv2.

### Training in Few-Shot Protocol

```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn few_shot \
data=few_shot_${dataset_name} shot=${shot} output=${your_ckpt_saving_path} trainer=${trainer_name}

# Example:
torchrun --nproc_per_node=4 main.py -cn few_shot \
data=few_shot_ssv2 shot=2 output=ckpt/few_shot_ssv2_2shot_tc_clip trainer=tc_clip
```
Here, `${shot}` represents the number of shots in K-shot training where K can be 2, 4, 8, or 16.

### Evaluation in Few-Shot Protocol

In this protocol, the final testing is not explicitly called, and the best validation performance during training is used as the final evaluation result.
However, if you wish to evaluate with a specific checkpoint after training, you can manually call:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn few_shot \
data=few_shot_${dataset_name} shot=${shot} output=${your_result_saving_path} trainer=${trainer_name} \
eval=val resume=${ckpt_path}

# Example:
torchrun --nproc_per_node=4 main.py -cn few_shot \
data=few_shot_ssv2 shot=2 output=/PATH/TO/OUTPUT trainer=tc_clip \
eval=val resume=ckpt/few_shot_ssv2_2shot_tc_clip/best.pth
```
Note that `eval=val` is specified to perform evaluation on the validation set using the saved checkpoint.



## Base-to-Novel Protocol
In the base-to-novel protocol, models are directly trained and evaluated on downstream datasets, including K-400, HMDB-51, UCF-101, and SSv2.
The training is conducted for three seeds for base classes,
and the harmonic mean of the average base accuracy and the average novel accuracy is reported.

### Training in Base-to-Novel Protocol

```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_${dataset_name} base=${base} output=${your_ckpt_saving_path} trainer=${trainer_name}

# Example:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_ssv2 base=1 output=ckpt/base2novel_ssv2_base1_tc_clip trainer=tc_clip
```
Note:
- `${base}` represents the seed number of base class where seed={1, 2, 3}.
- The best validation performance during training is reported as the top-1 accuracy on base classes.
- After training, `main_testing` is called to perform evaluation on novel categories with the best checkpoint.

### Evaluation in Base-to-Novel Protocol

Below are examples of manually calling evaluations from the saved checkpoint:
- For base classes:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_${dataset_name} base=${base} output=${your_result_saving_path} trainer=${trainer_name} \
eval=val resume=${ckpt_path}

# Example:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_ssv2 base=1 output=PATH/TO/OUTPUT trainer=tc_clip \
eval=val resume=ckpt/base2novel_ssv2_base1_tc_clip.pth
```
- For novel classes:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_${dataset_name} base=${base} output=${your_result_saving_path} trainer=${trainer_name} \
eval=test resume=${ckpt_path}

# Example:
torchrun --nproc_per_node=4 main.py -cn base2novel \
data=base2novel_ssv2 base=1 output=PATH/TO/OUTPUT trainer=tc_clip \
eval=test resume=ckpt/base2novel_ssv2_base1_tc_clip.pth
```
- Note the use of `eval=val` for base classes and `eval=test` for novel classes.


## Fully-Supervised Protocol
In this protocol, models are trained and evaluated using the full train/validation data of Kinetics-400 dataset.

### Training in Fully-Supervised Protocol

```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn fully_supervised \
data=fully_supervised_${dataset_name} output=${your_ckpt_saving_path} trainer=${trainer_name}

# Example:
torchrun --nproc_per_node=4 main.py -cn fully_supervised \
data=base2novel_k400 output=ckpt/fully_supervised_k400_tc_clip trainer=tc_clip
```
At the end of the training, `main_testing` function is called to perform multi-view inference on the validation dataset.

### Evaluation in Fully-Supervised Protocol

After training is finished, you also have the option to manually evaluate with the saved checkpoint using the following command:
```bash
# Basic usage:
torchrun --nproc_per_node=4 main.py -cn fully_supervised \
data=fully_supervised_${dataset_name} output=${your_result_saving_path} trainer=${trainer_name} \
eval=test resume=${ckpt_path}

# Example:
torchrun --nproc_per_node=4 main.py -cn fully_supervised \
data=fully_supervised_k400 output=/PATH/TO/OUTPUT trainer=tc_clip \
eval=test resume=ckpt/fully_supervised_k400_tc_clip/best.pth
```
