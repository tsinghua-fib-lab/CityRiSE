#!/bin/bash
# REMINDER: this script uses test data split and should ONLY be used for debugging. DO NOT use for training.

set -x

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=7,1,2,3

NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Detected NUM_GPU = $NUM_GPU"

MODEL_PATH=/init_ckpt/Qwen/Qwen2___5-VL-7B-Instruct  # replace it with your local file path


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/EasyR1/data/journeybench/train-00000-of-00001.parquet \
    data.val_files=/EasyR1/data/journeybench/test-00000-of-00001.parquet \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.limit_images=2 \
    trainer.experiment_name=qwen2_5_vl_7b_journeybench_multi_image_4gpu \
    trainer.n_gpus_per_node=$NUM_GPU
