#!/bin/bash

cd ../../

export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

ALGO=$1
MODEL=$2
USE_WANDB=$3
GN=$4

if [ -z "$ALGO" ]; then # if ALGO is not provided, set default
    ALGO="DDPG"
fi
if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL=null
fi
if [ -z "$USE_WANDB" ]; then # if ALGO is not provided, set default
    USE_WANDB=false
fi
if [ -z "$GN" ]; then # if ALGO is not provided, set default
    GN="null"
fi

ENVS=robotics
ENV_NAME=FetchReachDense-v4

max_repetition=10
total_training_steps=50000
eval_episodes=5
eval_interval=$((total_training_steps / 100))
eval_log_interval=$((total_training_steps / 4))
e_greedy_type=exponential
e_decay=$((total_training_steps / 10))

buffer_size=$total_training_steps
rep_buffer_size=$((total_training_steps*max_repetition))

video_save_dir=./videos/${ENV_NAME}/${ALGO}/${MODEL}/${GN}/

use_step_rate=true
hidden_dim=256
lr=0.001
use_lr_decay=true

use_geo_e_greedy=false
geo_p=0.3
alpha=0.01

use_adaptive_uncertainty=true

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
fi
if [ "$MODEL" != "EQL" ]; then
    use_geo_e_greedy=false
fi

if [ "$MODEL" == "UTE" ]; then
    EXTRA_ARGS+=("algos.models.use_adaptive_uncertainty=$use_adaptive_uncertainty")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
fi

for SEED in 0 1 2 3 4 5 6 7 8 9;
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "total_training_steps=$total_training_steps"
        "eval_interval=$eval_interval"
        "eval_episodes=$eval_episodes"
        "use_step_rate=$use_step_rate"
        "common_args.lr=$lr"
        "common_args.use_lr_decay=$use_lr_decay"
        "common_args.e_greedy_type=$e_greedy_type"
        "common_args.e_decay=$e_decay"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.buffer_size=$buffer_size"
        "common_args.rep_buffer_size=$rep_buffer_size"
        "common_args.use_geo_e_greedy=$use_geo_e_greedy"
        "common_args.geo_p=$geo_p"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done