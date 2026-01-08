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

ENVS=atari
ENV_NAME=MsPacmanNoFrameskip-v4

max_repetition=10
total_training_steps=1000000

buffer_size=$((total_training_steps / 3))
rep_buffer_size=$((total_training_steps*max_repetition / 3))
eval_interval=$((total_training_steps / 100))
eval_log_interval=$((total_training_steps / 4))
e_decay=$((total_training_steps / 5))

video_save_dir=null #./videos/${ENV_NAME}/${ALGO}/${MODEL}/${GN}/

use_dueling=true
use_image=true
lr=0.005
min_epsilon=0.05
use_hard_update=false
hidden_dim=512
batch_size=64

use_geo_e_greedy=false
geo_p=0.3
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
    EXTRA_ARGS+=("algos.models.max_repetition=$max_repetition")
fi
if [ "$ALGO" == "Random" ]; then
    total_training_steps=1
    eval_episodes=1000
fi
if [ "$MODEL" == "EQL" ]; then
    EXTRA_ARGS+=("algos.models.alpha=$alpha")
    EXTRA_ARGS+=("algos.models.fixed_coeff=$fixed_coeff")
fi
if [ "$MODEL" != "EQL" ]; then
    use_geo_e_greedy=false
fi

if [ "$MODEL" == "UTE" ]; then
    EXTRA_ARGS+=("algos.models.uncertainty_factor=$uncertainty_factor")
fi
if [ "$ENVS" != "null" ]; then
    EXTRA_ARGS+=("envs=$ENVS")
    EXTRA_ARGS+=("envs.env_name=$ENV_NAME")
fi
    
for SEED in 0 1 2 3 4
do
    ARGS=(
        "use_wandb=$USE_WANDB"
        "group_name=$GN"
        "seed=$SEED"
        "total_training_steps=$total_training_steps"
        "eval_interval=$eval_interval"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.buffer_size=$buffer_size"
        "common_args.rep_buffer_size=$rep_buffer_size"
        "common_args.use_geo_e_greedy=$use_geo_e_greedy"
        "common_args.use_image=$use_image"
        "common_args.use_dueling=$use_dueling"
        "common_args.e_decay=$e_decay"
        "common_args.min_epsilon=$min_epsilon"
        "envs.noop_max=30"
        "envs.frame_stack=4"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done