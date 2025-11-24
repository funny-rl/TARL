#!/bin/bash

export HYDRA_FULL_ERROR=1

cd ../

MODEL=$1
use_wandb=$2
gn=$3

if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL="DQN"
fi
if [ -z "$use_wandb" ]; then # if use_wandb is not provided, set default
    use_wandb=false
fi
if [ -z "$gn" ]; then # if group name is not provided, set default
    gn="default_group"
fi

total_training_steps=200000
eval_interval=1000
hidden_dim=128
e_greedy_type=power
use_step_rate=true

EXTRA_ARGS=()

if [ "$MODEL" = "GTA" ]; then
    EXTRA_ARGS+=("algos.alpha=0.8")
fi

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    ARGS=(
        "env_name=MountainCar-v0"
        "algos=$MODEL"
        "total_training_steps=$total_training_steps"
        "use_wandb=$use_wandb"
        "group_name=$gn"
        "seed=$SEED"
        "eval_interval=$eval_interval"
        "algos.hidden_dim=$hidden_dim"
        "e_greedy_type=$e_greedy_type"
        "use_step_rate=$use_step_rate"
        "${EXTRA_ARGS[@]}"
    )

    python main.py "${ARGS[@]}"
done