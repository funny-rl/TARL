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
ENV_NAME=PointMaze_Medium_Diverse_GRDense-v3
max_repetition=25
total_training_steps=400000
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
lr=0.0001
use_lr_decay=false
use_adaptive_uncertainty=true

alpha=0.7
n_sample=10
fixed_coeff=false
low_variance=true
use_act_skip_buf=false

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
    eval_episodes=100
fi
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
    EXTRA_ARGS+=("algos.models.max_repetition=$max_repetition")
fi
if [ "$MODEL" == "EQL" ]; then
    EXTRA_ARGS+=("algos.models.alpha=$alpha")
    EXTRA_ARGS+=("algos.models.fixed_coeff=$fixed_coeff")
    EXTRA_ARGS+=("algos.models.low_variance=$low_variance")
    EXTRA_ARGS+=("algos.models.continuous.n_sample=$n_sample")
fi
if [ "$MODEL" == "UTE" ]; then
    EXTRA_ARGS+=("algos.models.use_adaptive_uncertainty=$use_adaptive_uncertainty")
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
        "common_args.lr=$lr"
        "common_args.use_lr_decay=$use_lr_decay"
        "common_args.e_greedy_type=$e_greedy_type"
        "common_args.e_decay=$e_decay"
        "common_args.hidden_dim=$hidden_dim"
        "common_args.buffer_size=$buffer_size"
        "common_args.rep_buffer_size=$rep_buffer_size"
        "common_args.use_act_skip_buf=$use_act_skip_buf"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done