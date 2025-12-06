#!/bin/bash

cd ../
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

ENVS=atari
ENV_NAME=PongNoFrameskip-v4
total_training_steps=1000000
eval_interval=1000
video_save_dir=null #./videos/${ENV_NAME}/algo_${ALGO}/model_${MODEL}/

use_ddqn=true
use_dueling=true

EXTRA_ARGS=()

if [ "$video_save_dir" != "null" ]; then
    EXTRA_ARGS+=("video_save_dir='${video_save_dir}'")
fi
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("algos=$ALGO")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
fi
    
for SEED in 0
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "total_training_steps=$total_training_steps"
        "eval_interval=$eval_interval"
        "algos.use_ddqn=$use_ddqn"
        "algos.use_dueling=$use_dueling"
        "envs.noop_max=30"
        "envs.frame_stack=4"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done