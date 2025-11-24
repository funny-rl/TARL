#!/bin/bash

export HYDRA_FULL_ERROR=1

cd ../
ALGO=$1
MODEL=$2
use_wandb=$3
gn=$4

if [ -z "$ALGO" ]; then # if ALGO is not provided, set default
    ALGO="DQN"
fi
if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL=null
fi
if [ -z "$use_wandb" ]; then # if use_wandb is not provided, set default
    use_wandb=false
fi
if [ -z "$gn" ]; then # if group name is not provided, set default
    gn="default_group"
fi
total_training_steps=50000
eval_interval=2000
hidden_dim=128

EXTRA_ARGS=()

if [ "$MODEL" != "null" ]; then
    EXTRA_ARGS+=("algos/model=$MODEL")
    EXTRA_ARGS+=("algos.model.max_repetitions=20")
fi

for SEED in 0 #1 2 3 4 5 6 7 8 9
do
    ARGS=(
        "env_name=Pendulum-v1"
        "algos=$ALGO"
        "total_training_steps=$total_training_steps"
        "use_wandb=$use_wandb"
        "group_name=$gn"
        "seed=$SEED"
        "eval_interval=$eval_interval"
        "algo_args.hidden_dim=$hidden_dim"
        "${EXTRA_ARGS[@]}"
    )

    python main.py "${ARGS[@]}"
done