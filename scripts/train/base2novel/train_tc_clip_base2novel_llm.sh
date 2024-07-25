# base2novel training (llm version) with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

protocol=base2novel
dataset_name=k400_llm # choose one of {k400_llm, hmdb51_llm, ucf101_llm, ssv2_llm}
data=${protocol}_${dataset_name}

expr_name=tc_clip_reproduce
trainer=tc_clip
use_wandb=true

context_length=77

# base=1/2/3 training
for base in 1 2 3
do
  torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
  data=${data} \
  base=${base} \
  output=workspace/expr/${protocol}/${expr_name}/${protocol}_${dataset_name}_base${base}_${expr_name}_${trainer} \
  trainer=${trainer} \
  use_wandb=${use_wandb} \
  context_length=${context_length}
done