# TC-CLIP eval example with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

protocol=zero_shot
dataset_name=k400
data=${protocol}_${dataset_name}
resume=/PATH/TO/TRAINED/MODELS/zero_shot_k400_tc_clip.pth
trainer=tc_clip

wise_ft=0.7

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py \
data=${data} \
eval=test \
output=workspace/results/${data}/${data}_${trainer}_wise_ft_w${wise_ft} \
resume=${resume} \
trainer=${trainer} \
wise_ft=${wise_ft}