#!/bin/bash

cd ../../

export HYDRA_FULL_ERROR=1

ALGO=$1
MODEL=$2
USE_WANDB=$3
GN=$4


if [ -z "$ALGO" ]; then # if ALGO is not provided, set default
    ALGO="DQN"
fi
if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL=null
fi
if [ -z "$USE_WANDB" ]; then # if ALGO is not provided, set default
    USE_WANDB=false
fi
if [ -z "$GN" ]; then # if ALGO is not provided, set default
    GN="DQN"
fi

ENVS=grid
ENV_NAME=CliffWalking
max_repetition=9
total_training_steps=50000

buffer_size=$total_training_steps
rep_buffer_size=$((total_training_steps*max_repetition))

eval_interval=$((total_training_steps / 100))
eval_log_interval=$((total_training_steps / 4))
e_decay=$((total_training_steps * 9 / 10))

video_save_dir=./videos/${ENV_NAME}/${ALGO}/${MODEL}/${GN}/

hidden_dim=64
alpha=0.01
fixed_coeff=true
uncertainty_factor=-1.5
use_act_skip_buf=false
prev_buffer_save=false

EXTRA_ARGS=()

if [ "$video_save_dir" != "null" ]; then
    EXTRA_ARGS+=("video_save_dir='${video_save_dir}'")
    EXTRA_ARGS+=("log_eval_interval=$eval_log_interval")
fi
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("algos=$ALGO")
fi
if [ "$ALGO" == "Random" ]; then
    total_training_steps=1
    eval_episodes=1000
fi
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
    EXTRA_ARGS+=("algos.models.max_repetition=$max_repetition")
fi

if [ "$MODEL" == "EQL" ]; then
    EXTRA_ARGS+=("algos.models.alpha=$alpha")
    EXTRA_ARGS+=("algos.models.fixed_coeff=$fixed_coeff")
fi

if [ "$MODEL" == "UTE" ]; then
    EXTRA_ARGS+=("algos.models.uncertainty_factor=$uncertainty_factor")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
fi

for SEED in 10 20 30 40 50 60 70 80 90 100;
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "total_training_steps=$total_training_steps"
        "eval_interval=$eval_interval"
        "common_args.e_decay=$e_decay"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.buffer_size=$buffer_size"
        "common_args.rep_buffer_size=$rep_buffer_size"
        "common_args.use_act_skip_buf=$use_act_skip_buf"
        "common_args.prev_buffer_save=$prev_buffer_save"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done