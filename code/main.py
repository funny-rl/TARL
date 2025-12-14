import json
import os
import imageio
import time
from matplotlib.pyplot import _log
import wandb
import hydra
import torch
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils.utils import (
    set_seed, 
    episode_stats, 
    state_transform,
    action_transform
)
from utils.build_env import build_env, get_env_info
from utils.set_hyperparam import set_hyperparam

from algos import ALGO_REGISTRY, MODEL_REGISTRY

@hydra.main(config_path="configs/", config_name="config", version_base=None)
def main(args):
    print("-"*20, "Experiment Configuration", "-"*20)
    print(OmegaConf.to_yaml(args))
    print("-"*60)

    algo_args = args.algos
    env_args = args.get("envs", None)
    model_args = algo_args.get("models", None)
    common_args = args.common_args
    
    use_wandb: bool = args.use_wandb
    use_step_rate: bool = args.use_step_rate
    use_lr_decay: bool = common_args.use_lr_decay
    use_eval_render: bool = True if args.video_save_dir is not None else False
    save_model: bool = True if args.save_dir is not None else False
    
    algo_name: str = algo_args.algo_name    
    env_name: str = env_args.env_name if env_args is not None else None
    video_save_dir: str = args.video_save_dir
    model_name: str = model_args.model_name if model_args is not None else None
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed: int = args.seed
    training_steps: int = 0
    num_episodes: int = 0
    save_interval: int = args.save_interval 
    eval_interval: int = args.eval_interval
    eval_episodes: int = args.eval_episodes
    eval_max_steps: int = args.eval_max_steps
    warmup_steps: int = args.warmup_steps
    total_training_steps: int = args.total_training_steps
    log_eval_interval: int = args.log_eval_interval

    set_seed(seed)
    
    env = build_env(env_args)
    env_info = get_env_info(env_args, env, use_step_rate)
    
    OmegaConf.set_struct(common_args, False)
    OmegaConf.set_struct(algo_args, False)
    hyper_args: DictConfig = OmegaConf.merge(
        common_args, 
        algo_args
    )
    base_config, model_config = set_hyperparam(
        hyper_args, 
        model_args, 
        env_info, 
        device
    )
    
    base_agent = ALGO_REGISTRY[algo_name](**base_config)
    
    if model_name is None:
        rep_agent = base_agent
    else:
        rep_agent = MODEL_REGISTRY[model_name](base_agent, **model_config)
        
    if use_wandb:
        wandb.init(
            project=env_name, 
            name=f"{algo_name}_{model_name}_{env_name}_{seed}",
            group=args.group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        
    state, _ =  env.reset()
    
    episode_stats_dict: dict[str, Any] = {
        "episode_reward": 0.0,
        "repetition": [],
        "clipped_actor_grad_norm": [],
        "td_error": [],
        "q_loss": [],
        "rep_q_loss": [],
        "rep_td_error": [],
        "actor_loss": [],
        "critic_loss": [],
    }
    
    while training_steps <= total_training_steps:
        with episode_stats(episode_stats_dict) as log: 
            done = False
            state = state_transform(state, use_step_rate, env, device)
            while not done:
                train = training_steps >= warmup_steps
                if train:
                    action = base_agent.select_action(state)
                    repetition = rep_agent.select_repetition(
                        state,
                        action_transform(
                            action, 
                            env_info.get("n_actions", None), 
                            device
                        ),
                    )
                else:
                    action = env.action_space.sample()
                    repetition = rep_agent.select_repetition(state)

                log["repetition"].append(repetition)
                
                skip_states, skip_rewards = [], []
                for _ in range(repetition):
                    (
                        next_state, 
                        reward, 
                        terminated, 
                        truncated, 
                        _
                    ) = env.step(action)
                    done: bool = terminated or truncated
                    next_state = state_transform(next_state, use_step_rate, env, device)
                    log["episode_reward"] += float(reward)
                    skip_states.append(state)
                    skip_rewards.append(reward)
                    
                    rep_agent.add(
                        state.cpu(),
                        action,
                        reward,
                        next_state.cpu(),
                        done,
                        skip_states,
                        skip_rewards
                    )
                    state = next_state
                    
                    if train:
                        log_dict: dict[str, Any] = rep_agent.update(training_steps)
                        for key, value in log_dict.items():
                            log[key].append(value)

                    if eval_interval > 0 and training_steps % eval_interval == 0:
                        eval(
                            env_name = env_name,
                            env_args = env_args,
                            rep_agent = rep_agent,
                            seed = seed,
                            eval_episodes = eval_episodes,
                            eval_max_steps = eval_max_steps,
                            use_step_rate = use_step_rate,
                            training_steps = training_steps,
                            use_wandb = use_wandb,
                            use_eval_render = use_eval_render,
                            video_save_dir = video_save_dir,
                            log_eval_interval = log_eval_interval,
                            env_info = env_info,
                            device = device
                        )
                    
                    if save_model and training_steps % save_interval == 0:
                        save_dir = os.path.join(args.save_dir, f"{algo_name}_{model_name}", f"{training_steps}/")
                        os.makedirs(save_dir, exist_ok=True)
                        rep_agent.save_model(save_dir)
                        print(f"Saved model at {save_dir}")

                    if hasattr(rep_agent, "epsilon_decay"):
                        rep_agent.epsilon_decay(
                            max(training_steps - warmup_steps, 0.0),
                        )
                    if use_lr_decay:
                        training_rate = max(
                            0.0, 
                            min(
                                1.0, 
                                (training_steps - warmup_steps) / (total_training_steps - warmup_steps)
                            )
                        )
                        rep_agent.lr_decay(training_rate)

                    if done:
                        num_episodes += 1
                        state, _ = env.reset()
                        log["lr"] = rep_agent.lr
                        if hasattr(rep_agent, "expl_alpha"):
                            log["alpha"] = rep_agent.expl_alpha
                        
                        msg = f"Training steps: {training_steps} | Episode: {num_episodes} | Rewards: {log['episode_reward']} | LR: {rep_agent.lr} "
                        if hasattr(rep_agent, "epsilon"):
                            log["epsilon"] = rep_agent.epsilon
                            msg += f"| Epsilon: {rep_agent.epsilon}"
                        print(msg)
                        if use_wandb:
                            _log = {}
                            for key, value in log.items():
                                try:
                                    if isinstance(value, list) and len(value) > 0:
                                        _log[f"train/{key}"] = np.mean(value)
                                    elif isinstance(value, (float, int)):
                                        _log[f"train/{key}"] = value
                                except:
                                    raise ValueError(f"Invalid log value type: {type(value)} for key: {key}")
                            wandb.log(_log, step=training_steps)
                        break
                    training_steps += 1
    env.close()

def eval(
    env_name,
    env_args,
    rep_agent,
    seed,
    eval_episodes,
    eval_max_steps,
    use_step_rate,
    training_steps,
    use_wandb,
    use_eval_render,
    video_save_dir,
    log_eval_interval,
    env_info,
    device
):
    best_frames = None
    best_log = None
    best_reward = float("-inf")  
    
    print("\n\nStarting Evaluation...\n")
    eval_env = build_env(
        env_args, 
        use_eval_render
    )

    total_rewards: list[float] = []
    eval_repetition: list[int] = []
    
    for ep in range(eval_episodes):
        start_time = time.time()
        done = False
        frames: list[Any] = []
        episode_reward: float = 0.0
        epi_repetition: list[int] = []
        state, _ = eval_env.reset(seed=seed + 100 * ep)
        eval_step = 0
        eval_log: list[dict[str, Any]] = []

        while not done:
            state = state_transform(state, use_step_rate, eval_env, device)
            action = rep_agent.select_action(state, deterministic=True)
            repetition, rep_Qs = rep_agent.select_repetition(
                state,
                action_transform(
                    action, 
                    env_info.get("n_actions", None), 
                    device
                ),
                deterministic=True
            )
            epi_repetition.append(repetition)

            eval_log.append({
                "step": eval_step,
                "state": state.cpu().numpy().tolist(),
                "action": action.tolist(),
                "repetition": repetition,
                "rep_Qs": rep_Qs.flatten().cpu().numpy().tolist() if rep_Qs is not None else None,
            })
            
            for _ in range(repetition):
                eval_step += 1
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                if use_eval_render and (training_steps % log_eval_interval == 0):
                    frame = eval_env.render()
                    frames.append(frame)
                state = next_state
                done: bool = terminated or truncated
                episode_reward += reward
                if done:
                    break
                if eval_max_steps < eval_step:
                    done = True
                    break

        end_time = time.time()
        eval_log.append({"episode_reward": episode_reward})
        print(f"[Evaluation] Episode: {ep+1} | Reward: {episode_reward} | Time: {end_time - start_time:.2f} seconds")
        if (use_eval_render and (training_steps % log_eval_interval == 0)) and episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames.copy()
            best_log = eval_log.copy()
        total_rewards.append(episode_reward)
        eval_repetition.append(np.mean(epi_repetition))
    
    if use_eval_render and (training_steps % log_eval_interval == 0):
        save_dir = os.path.join(video_save_dir, str(training_steps))
        os.makedirs(save_dir, exist_ok=True)
        video_path = f"{save_dir}/test.mp4"
        
        imageio.mimsave(video_path, best_frames, fps=30)
        print(f"Saved evaluation video at {video_path}")

        log_path = f"{save_dir}/eval_log.json"
        with open(log_path, 'w') as f:
            json.dump(best_log, f, indent=4)
            print(f"Saved evaluation log at {log_path}")

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
    print(f"[Evaluation] Training steps: {training_steps} | Average Reward: {avg_reward} | Average Repetition: {avg_repetition} | Std Repetition: {std_repetition}")
    eval_env.close()
    
if __name__ == "__main__":
    main()