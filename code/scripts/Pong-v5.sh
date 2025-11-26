#!/bin/bash

export HYDRA_FULL_ERROR=1

cd ../

ALGO=$1
MODEL=$2
USE_WANDB=$3
GN=$4

if [ -z "$ALGO" ]; then # if ALGO is not provided, set default
    ALGO="DQN"
fi
if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL="null"
fi
if [ -z "$USE_WANDB" ]; then # if USE_WANDB is not provided, set default
    USE_WANDB="false"
fi
if [ -z "$GN" ]; then # if GN is not provided, set default
    GN="default"
fi

total_training_steps=200000
warmup_steps=5000
eval_interval=5000
hidden_dim=512
e_greedy_type=linear
use_step_rate=false
use_lr_decay=false
data_type="image"
batch_size=128
lr=0.0005

EXTRA_ARGS=()

EXTRA_ARGS+=("envs=atari")
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
fi

for SEED in 0
do
    ARGS=(
        "env_name=ALE/Pong-v5"
        "algos=$ALGO"
        "total_training_steps=$total_training_steps"
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "algo_args.hidden_dim=$hidden_dim"
        "algo_args.e_greedy_type=$e_greedy_type"
        "use_step_rate=$use_step_rate"
        "use_lr_decay=$use_lr_decay"
        "eval_interval=$eval_interval"
        "algo_args.lr=$lr"
        "algo_args.data_type=$data_type"
        "algo_args.batch_size=$batch_size"
        "warmup_steps=$warmup_steps"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done