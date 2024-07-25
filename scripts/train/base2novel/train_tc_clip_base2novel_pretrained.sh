# base2novel training (using pretrained model) with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

protocol=base2novel
dataset_name=k400 # choose one of {hmdb51, ucf101, ssv2}
data=${protocol}_${dataset_name}

expr_name=tc_clip_reproduce
trainer=tc_clip
use_wandb=true

resume=/PATH/TO/PRETRAINED/MODELS/zero_shot_k400_tc_clip.pth

# base=1/2/3 training
for base in 1 2 3
do
  torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
  data=${data} \
  base=${base} \
  output=workspace/expr/${protocol}/${expr_name}/${protocol}_${dataset_name}_base${base}_${expr_name}_${trainer} \
  trainer=${trainer} \
  use_wandb=${use_wandb} \
  resume=${resume}
done