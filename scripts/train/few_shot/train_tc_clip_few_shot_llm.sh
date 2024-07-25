# few-shot training (llm version) with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

protocol=few_shot
dataset_name=hmdb51_llm # choose one of {hmdb51_llm, ucf101_llm}
data=${protocol}_${dataset_name}

expr_name=tc_clip_reproduce
trainer=tc_clip
use_wandb=true

context_length=77

# k-shot training
for shot in 2 4 8 16
do
  torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
  data=${data} \
  shot=${shot} \
  output=workspace/expr/${protocol}/${expr_name}/${protocol}_${dataset_name}_${shot}shot_${expr_name}_${trainer} \
  trainer=${trainer} \
  use_wandb=${use_wandb} \
  context_length=${context_length}
done