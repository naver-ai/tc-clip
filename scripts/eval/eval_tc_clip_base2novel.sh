# TC-CLIP base2novel eval example with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

protocol=base2novel
dataset_name=k400 # choose one of {k400, hmdb51, ucf101, ssv2}
data=${protocol}_${dataset_name}
base=1  # choose one of {1, 2, 3}
resume=/PATH/TO/TRAINED/MODELS/${protocol}_${dataset_name}_base${base}_tc_clip.pth
trainer=tc_clip

# base
torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
base=${base} \
eval=val \
output=workspace/results/${data}/${data}_base${base}_${trainer} \
trainer=${trainer} \
resume=${resume}

# novel
torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
eval=test \
output=workspace/results/${data}/${data}_novel_${trainer} \
trainer=${trainer} \
resume=${resume}