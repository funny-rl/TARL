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
    GN="DDPG"
fi

ENVS=robotics
ENV_NAME=PointMaze_Medium_Diverse_GRDense-v3
total_training_steps=500000
eval_interval=5000
eval_episodes=10
video_save_dir=./videos/${ENV_NAME}/algo_${ALGO}/model_${MODEL}/
use_dueling=false
e_decay=30000 #  standard: total_training_steps * 0.1
use_step_rate=true
use_image=false
lr=0.0005
use_lr_decay=false
use_hard_update=true
hidden_dim=512
buffer_size=100000
batch_size=64
use_geo_e_greedy=true
geo_p=0.5
alpha=0.01
uncertainty_factor=-2.0

EXTRA_ARGS=()

if [ "$video_save_dir" != "null" ]; then
    EXTRA_ARGS+=("video_save_dir='${video_save_dir}'")
fi
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("algos=$ALGO")
fi
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
    EXTRA_ARGS+=("algos.models.max_repetition=25")
fi
if [ "$MODEL" == "EQL" ]; then
    EXTRA_ARGS+=("algos.models.alpha=$alpha")
fi
if [ "$MODEL" == "UTE" ]; then
    EXTRA_ARGS+=("algos.models.uncertainty_factor=$uncertainty_factor")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
    EXTRA_ARGS+=("envs.max_episode_steps=500")
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
        "common_args.use_dueling=$use_dueling"
        "common_args.e_decay=$e_decay"
        "common_args.use_image=$use_image"
        "common_args.lr=$lr"
        "common_args.use_lr_decay=$use_lr_decay"
        "common_args.use_hard_update=$use_hard_update"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.use_geo_e_greedy=$use_geo_e_greedy"
        "common_args.buffer_size=$buffer_size"
        "common_args.batch_size=$batch_size"
        "common_args.use_geo_e_greedy=$use_geo_e_greedy"
        "common_args.geo_p=$geo_p"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done