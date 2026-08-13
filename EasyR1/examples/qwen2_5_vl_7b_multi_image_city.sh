#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4,5,6,7


NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


echo "Detected NUM_GPU = $NUM_GPU"

MODEL_PATH=init_ckpt/Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=data/training/RL/mix-data-v2.9.parquet \
    data.val_files=data/training/RL/mix-data-v2.9.parquet \
    trainer.total_epochs=4 \
    data.max_prompt_length=120000 \
    data.rollout_batch_size=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=8 \
    worker.rollout.limit_images=11 \
    trainer.experiment_name=qwen2_5_vl_7b-mix-data-v2.9-huber-reward-epoch4 \
    trainer.n_gpus_per_node=$NUM_GPU \
    trainer.val_freq=-1  \
    trainer.val_before_train=false \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.offload.offload_params=true \
