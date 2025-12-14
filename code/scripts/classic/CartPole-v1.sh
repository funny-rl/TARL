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

ENVS=classic
ENV_NAME=CartPole-v1
total_training_steps=100000
eval_interval=500
eval_episodes=10
video_save_dir=null #./videos/${ENV_NAME}/algo_${ALGO}/model_${MODEL}/
use_dueling=false
e_decay=10000 #  standard: total_training_steps * 0.1
use_step_rate=true
use_image=false
lr=0.00025
use_lr_decay=true
use_hard_update=false
hidden_dim=64
use_geo_e_greedy=false

EXTRA_ARGS=()

if [ "$video_save_dir" != "null" ]; then
    EXTRA_ARGS+=("video_save_dir='${video_save_dir}'")
fi
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("algos=$ALGO")
fi
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
    EXTRA_ARGS+=("algos.models.max_repetition=5")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
fi
    
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "total_training_steps=$total_training_steps"
        "eval_interval=$eval_interval"
        "eval_episodes=$eval_episodes"
        "use_step_rate=$use_step_rate"
        "common_args.e_decay=$e_decay"
        "common_args.use_image=$use_image"
        "common_args.lr=$lr"
        "common_args.use_lr_decay=$use_lr_decay"
        "common_args.use_hard_update=$use_hard_update"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.use_dueling=$use_dueling"
        "common_args.use_geo_e_greedy=$use_geo_e_greedy"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done