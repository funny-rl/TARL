#!/bin/bash

export HYDRA_FULL_ERROR=1

cd ../

MODEL=$1
use_wandb=$2
gn=$3
use_rep_max_q=$4

if [ -z "$MODEL" ]; then # if MODEL is not provided, set default
    MODEL="Q_learning"
fi
if [ -z "$use_wandb" ]; then # if use_wandb is not provided, set default
    use_wandb=false
fi
if [ -z "$gn" ]; then # if group name is not provided, set default
    gn="default_group"
fi
if [ -z "$use_rep_max_q" ]; then # if use_rep_max_q is not provided, set default
    use_rep_max_q=true
fi

if [ "$MODEL" = "Q_learning" ] || [ "$MODEL" = "RQ_learning" ]; then
    total_training_steps=1000000
    eval_interval=10000
fi

if [ "$MODEL" = "DQN" ] || [ "$MODEL" = "TempoRL" ] || [ "$MODEL" = "UTE" ]; then
    total_training_steps=100000
    eval_interval=5000
    hidden_dim=64
fi

EXTRA_ARGS=()

if [ "$MODEL" = "RQ_learning" ] || [ "$MODEL" = "TempoRL" ]; then
    EXTRA_ARGS+=("algos.use_rep_max_q=$use_rep_max_q")
fi
if [ "$MODEL" = "DQN" ] || [ "$MODEL" = "TempoRL" ] || [ "$MODEL" = "UTE" ]; tWhen
    EXTRA_ARGS+=("algos.hidden_dim=$hidden_dim")
fi

for SEED in 0 1 2 3 4
do
    ARGS=(
        "env_name=CliffWalking-v1"
        "algos=$MODEL"
        "total_training_steps=$total_training_steps"
        "use_wandb=$use_wandb"
        "group_name=$gn"
        "seed=$SEED"
        "eval_interval=$eval_interval"
        "e_greedy_type=linear"
        "${EXTRA_ARGS[@]}"
    )

    python main.py "${ARGS[@]}"
done