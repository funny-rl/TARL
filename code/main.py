
import wandb
import hydra
import numpy as np
import gymnasium as gym

from omegaconf import OmegaConf

from typing import Any
from algos import ALGO_REGISTRY, MODEL_REGISTRY
from utils import (
    build_env, 
    set_seed, 
    get_model_configs, 
    episode_stats, 
    state_transform, 
    to_wandb_name
)

@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(args):
    print("-"*20, "Experiment Configuration", "-"*20)
    print(OmegaConf.to_yaml(args))
    print("-"*60)
    
    env_name: str = args.env_name
    
    use_wandb: bool = args.use_wandb
    use_step_rate: bool = args.use_step_rate
    use_lr_decay: bool = args.use_lr_decay
    
    num_episodes: int = 0
    seed: int = args.seed
    training_steps: int = 0
    warmup_steps: int = args.warmup_steps
    eval_episodes: int = args.eval_episodes
    eval_interval: int = args.eval_interval
    total_training_steps: int = args.total_training_steps
    
    set_seed(seed)
    env: gym.Env = build_env(env_name, use_step_rate, args)
    eval_env: gym.Env = build_env(env_name, use_step_rate, args)
    
    algo_dict, model_dict = get_model_configs(args)
    algo_name: str = args.algos.name
    
    action_agent = ALGO_REGISTRY[algo_name](**algo_dict)
    
    model_args = args.algos.get("models", None)
    if model_args is None:
        model_name = None  
        rep_agent = action_agent
    else:
        model_name = model_args.get("name", None)    
        rep_agent = MODEL_REGISTRY[model_name](action_agent, **model_dict)
    
    state, _ =  env.reset(seed = seed)
    
    if use_wandb:
        wandb.init(
            project=to_wandb_name(env_name),       
            name=f"{algo_name}_{model_name}_seed{args.seed}",
            group=args.group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
    
    episode_stats_dict: dict[str, Any] = {
        "episode_reward": 0.0,
        "td_error": [],
        "q_loss": [],
        "repetition": [],
    }
    
    while training_steps < total_training_steps:
        with episode_stats(episode_stats_dict) as _log:
            done = False
            state = state_transform(state, use_step_rate, env)
            while not done:
                train = training_steps >= warmup_steps
                if train:
                    action = action_agent.select_action(state)
                    repetition = rep_agent.select_repetition(
                        state,
                        action,
                    )
                else:
                    action = env.action_space.sample()
                    repetition = rep_agent.select_repetition(state)
                print(f"Selected action: {action}")
                _log["repetition"].append(repetition)
                
                skip_states, skip_rewards = [], []
                for _ in range(repetition):
                    training_steps += 1
                    (
                        next_state, 
                        reward, 
                        terminated, 
                        truncated, 
                        _
                    ) = env.step(action)
                    done: bool = terminated or truncated
                    next_state = state_transform(next_state, use_step_rate, env)
                    
                    _log["episode_reward"] += reward
                    skip_states.append(state)
                    skip_rewards.append(reward)
                    
                    rep_agent.add(
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        skip_states = skip_states,
                        skip_rewards = skip_rewards
                    )
                    if train:
                        log_dict: dict[str, Any] = rep_agent.update()
                        for key, value in log_dict.items():
                            _log[key].append(value)
                            
            
                    state = next_state
                    if eval_interval > 0 and training_steps % eval_interval == 0:
                        eval(
                            eval_env,
                            rep_agent,
                            seed,
                            eval_episodes,
                            use_step_rate,
                            training_steps,
                            use_wandb,
                        )
                        
                    training_rate = (training_steps - warmup_steps) / (total_training_steps - warmup_steps)
                    rep_agent.epsilon_decay(training_rate)
                    if use_lr_decay:
                        rep_agent.lr_decay(training_rate)
                        
                    if done:
                        num_episodes += 1
                        state, _ = env.reset(seed = seed + num_episodes)
                        _log["lr"] = rep_agent.lr
                        _log["epsilon"] = rep_agent.epsilon
                        print(f"Training steps: {training_steps} | Episode: {num_episodes} | LR: {rep_agent.lr} | Epsilon: {rep_agent.epsilon} | Rewards: {_log['episode_reward']}")
                        if use_wandb:
                            log = {}
                            for key, value in _log.items():
                                if isinstance(value, list) and len(value) > 0:
                                    log[f"train/{key}"] = np.mean(value)
                                elif isinstance(value, (float, int)):
                                    log[f"train/{key}"] = value
                            wandb.log(log, step=training_steps)
                        break
    env.close()
    eval_env.close()

def eval(
    eval_env: gym.Env,
    rep_agent,
    seed: int,
    eval_episodes: int,
    use_step_rate: bool,
    training_steps: int,
    use_wandb: bool,
):
    

    total_rewards: list[float] = []
    eval_repetition: list[int] = []
    
    for ep in range(eval_episodes):
        state, _ = eval_env.reset(seed = seed + ep + 1000)
        done = False
        episode_reward: float = 0.0
        epi_repetition: list[int] = []
        while not done:
            state = state_transform(state, use_step_rate, eval_env)
            action = rep_agent.select_action(
                state,
                deterministic = True
            )
            repetition = rep_agent.select_repetition(
                state,
                action,
                deterministic = True
            )
            epi_repetition.append(repetition)
            for _ in range(repetition):
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                if done:
                    break
        total_rewards.append(episode_reward)
        eval_repetition.append(np.mean(epi_repetition))

    avg_reward: float = np.mean(total_rewards)
    avg_repetition: float = np.mean(eval_repetition)
    std_repetition: float = np.std(eval_repetition)
    if use_wandb:
        wandb.log(
            {
                "eval/average_reward": avg_reward,
                "eval/average_repetition": avg_repetition,
                "eval/std_repetition": std_repetition,
            },
            step=training_steps,
        )
    print(f"[Evaluation] Training steps: {training_steps} | Average Reward over {eval_episodes} episodes: {avg_reward}")
    
    
if __name__ == "__main__":
    main()