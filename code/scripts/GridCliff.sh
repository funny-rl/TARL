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

ENVS=grid
ENV_NAME=grid_cliff
total_training_steps=150000
eval_interval=10000
eval_episodes=10
use_ddqn=false
use_dueling=true
e_decay=50000

EXTRA_ARGS=()

# if [ "$video_save_dir" != "null" ]; then
#    EXTRA_ARGS+=("video_save_dir='${video_save_dir}'")
# fi
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("algos=$ALGO")
fi
if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/models=$MODEL")
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
        "eval_episodes=$eval_episodes"
        "algos.use_ddqn=$use_ddqn"
        "algos.use_dueling=$use_dueling"
        "common_args.e_decay=$e_decay"
        "common_args.use_image=false"

        "+envs.pits='[[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7]]'"
        "+envs.shape='[6,10]'"
        "+envs.start='[0,0]'"
        "+envs.goal='[0,9]'"
        "+envs.max_steps=100"
        "${EXTRA_ARGS[@]}"
    )
    python main.py "${ARGS[@]}"
done