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
    action_transform,
    reward_transform
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
    use_act_skip_buf: bool = common_args.use_act_skip_buf
    use_eval_render: bool = True if args.video_save_dir is not None else False
    save_model: bool = args.save_model
    algo_name: str = algo_args.algo_name
    env_name: str = env_args.env_name if env_args is not None else None
    video_save_dir: str = args.video_save_dir
    model_name: str = model_args.model_name if model_args is not None else None
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_name = args.group_name
    
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
    display_eval_interval: int = args.display_eval_interval
    
    set_seed(seed)
    
    env = build_env(env_args)
    env_info = get_env_info(env_args, env, use_step_rate)

    use_log_reward: bool = env_info.get("use_log_reward", False)
    
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
        "rep_max_td_error": [],
        "rep_mean_td_error": [],
        "rep_mean_q_loss": [],
        "rep_max_q_loss": [],
    }
    
    has_base_agent: bool = hasattr(rep_agent, "base_agent")
    use_adaptive_lambda = getattr(rep_agent, "use_adaptive_uncertainty", False)
    
    if model_name == "TAAC":
        assert use_act_skip_buf == False, "TAAC model does not support action skip buffer."
        
    else:
        assert has_base_agent or not use_act_skip_buf, "Action skip buffer can be used only when there is a base agent."
    
    if use_adaptive_lambda:
        assert hasattr(rep_agent, "ucb"), "Adaptive repetition lambda requires UCB Algorithms."
    
    if use_wandb:
        unique_id = unique_id = f"{algo_name}_{model_name}_{group_name}_{env_name}_{seed}_{time.time()}"
        wandb.init(
            project=env_name, 
            id=unique_id,
            name=f"{algo_name}_{model_name}_{env_name}_{seed}",
            group=group_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
    
    while training_steps <= total_training_steps:
        with episode_stats(episode_stats_dict) as log: 
            done = False
            prev_action = None
            
            state = state_transform(state, use_step_rate, env, device)
            if use_adaptive_lambda:
                rep_agent.adaptive_lambda()
                
            while not done:
                train = training_steps >= warmup_steps
                
                if train:
                    if model_name == "TAAC":
                        action, beta = rep_agent.select_action(state, prev_action)
                    else:
                        action = rep_agent.select_action(state)
                        repetition = rep_agent.select_repetition(
                            state,
                            action_transform(
                                action, 
                                env_info.get("n_actions", None), 
                                device
                            ),
                        )
                else:
                    if model_name == "TAAC":
                        beta = 1.0

                    action = env.action_space.sample()
                    repetition = rep_agent.select_repetition(state)

                log["repetition"].append(repetition)
                (
                    skip_states, 
                    skip_rewards, 
                    skip_dones, 
                    next_skip_states
                ) = [], [], [], []

                if env_info.get("int_action", None) is not None:
                    action = int(action.item())
                for _ in range(repetition):
                    (
                        next_state, 
                        reward, 
                        terminated, 
                        truncated, 
                        _
                    ) = env.step(action)
                    reward = reward_transform(reward, use_log_reward)
                    done: bool = terminated or truncated
                    next_state = state_transform(next_state, use_step_rate, env, device)
                    log["episode_reward"] += float(reward)

                    skip_states.append(state)
                    skip_rewards.append(reward)
                    skip_dones.append(done)
                    next_skip_states.append(next_state)

                    if not use_act_skip_buf:
                        if has_base_agent:
                            rep_agent.base_agent.add(
                                state = state.cpu(),
                                action = action,
                                reward = reward,
                                next_state = next_state.cpu(),
                                done = done,
                            )
                        else:
                            if model_name == "TAAC":
                                rep_agent.add(
                                    state = state.cpu(),
                                    action = action,
                                    prev_action = prev_action,
                                    reward = reward,
                                    next_state = next_state.cpu(),
                                    beta = beta,
                                    done = done,
                                )
                            else:
                                rep_agent.add(
                                    state = state.cpu(),
                                    action = action,
                                    reward = reward,
                                    next_state = next_state.cpu(),
                                    done = done,
                                )

                    state = next_state
                    prev_action = action
                    
                    if train:
                        log_dict: dict[str, Any] = rep_agent.update(training_steps)
                        for key, value in log_dict.items():
                            log[key].append(value)

                    if eval_interval > 0 and training_steps % eval_interval == 0:
                        eval(
                            model_name = model_name,
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
                            display_eval_interval = display_eval_interval,
                            env_info = env_info,
                            device = device
                        )
                    
                    if save_model and training_steps % save_interval == 0:
                        save_dir = os.path.join(args.save_dir, f"{algo_name}_{model_name}", f"{training_steps}/")
                        os.makedirs(save_dir, exist_ok=True)
                        rep_agent.save_model(save_dir)
                        print(f"Saved model at {save_dir}")

                    if use_adaptive_lambda:
                        rep_agent.ucb_datas.append(
                            (
                                rep_agent.j, 
                                reward
                            )
                        )
                    
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

                        msg = f"Training steps: {training_steps} | Episode: {num_episodes} | Rewards: {log['episode_reward']} | LR: {rep_agent.lr} | {env_name} | Algo: {algo_name} | Model: {model_name} | Group: {group_name} | Seed: {seed}"
                        if hasattr(rep_agent, "epsilon"):
                            log["epsilon"] = rep_agent.epsilon
                            msg += f"| Epsilon: {rep_agent.epsilon}"

                        if hasattr(rep_agent, "alpha"):
                            log["alpha"] = rep_agent.alpha
                            msg += f"| Alpha: {rep_agent.alpha}"

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

                if has_base_agent:
                    rep_agent.add(
                        skip_states,
                        action,
                        skip_rewards,
                        skip_dones,
                        next_skip_states,
                        repetition
                    )

            if use_adaptive_lambda:
                rep_agent.ucb.push_data(rep_agent.ucb_datas)

    env.close()

def eval(
    model_name,
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
    display_eval_interval,
    env_info,
    device
): 

    use_log_reward = env_info.get("use_log_reward", False)
    
    print("\n\nStarting Evaluation...\n")
    eval_env = build_env(
        env_args, 
        use_eval_render
    )

    total_rewards: list[float] = []
    eval_repetition: list[int] = []
    eval_num_decision: list[int] = []  
    total_eval_time: list[float] = []
    for ep in range(eval_episodes):
        done = False
        frames: list[Any] = []
        episode_reward: float = 0.0
        num_decision: int = 0
        epi_repetition: list[int] = []
        test_seed=seed + 100 * ep
        state, _ = eval_env.reset(seed = test_seed)
        eval_step = 0
        eval_log: list[dict[str, Any]] = []
        prev_action = None
        
        start_time = time.time()

        while not done:
            state = state_transform(state, use_step_rate, eval_env, device)
            
            if model_name == "TAAC":
                action, _ = rep_agent.select_action(state, prev_action, deterministic=True)
            else:
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
            num_decision += 1
            
            eval_log.append({
                "step": eval_step,
                "state": state.cpu().numpy().tolist(),
                "action": action.tolist(),
                "repetition": repetition,
                "rep_Qs": rep_Qs.flatten().cpu().numpy().tolist() if rep_Qs is not None else None,
            })
            
            for _ in range(repetition):
                eval_step += 1
                if env_info.get("int_action", None) is not None:
                    action = int(action.item())
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                reward = reward_transform(reward, use_log_reward)
                done: bool = terminated or truncated

                if use_eval_render and (training_steps % display_eval_interval == 0):
                    frame = eval_env.render()
                    frames.append(frame)
                
                state = next_state
                prev_action = action
                
                episode_reward += reward
                if done:
                    break
                if eval_max_steps < eval_step:
                    done = True
                    break

        end_time = time.time()
        eval_log.append({"episode_reward": episode_reward})
        print(f"[Evaluation] Episode: {ep+1} | Reward: {episode_reward} | Time: {end_time - start_time:.2f} seconds")
        
        total_eval_time.append(end_time - start_time)
        total_rewards.append(episode_reward)
        eval_repetition.append(np.mean(epi_repetition))
        eval_num_decision.append(num_decision)
        
    
    save_dir = os.path.join(video_save_dir, str(training_steps))
    if use_eval_render and (training_steps % log_eval_interval == 0):
        os.makedirs(save_dir, exist_ok=True)
        log_path = f"{save_dir}/{test_seed}/eval_log.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(eval_log, f, indent=4)
            print(f"Saved evaluation log at {log_path}")

    
    if use_eval_render and (training_steps % display_eval_interval == 0):
        video_path = f"{save_dir}/{test_seed}/test.mp4"
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved evaluation video at {video_path}")

    avg_reward: float = np.mean(total_rewards)
    avg_repetition: float = np.mean(eval_repetition)
    std_repetition: float = np.std(eval_repetition)
    avg_decision: float = np.mean(eval_num_decision)
    avg_eval_time: float = np.mean(total_eval_time)
    
    if use_wandb:
        wandb.log(
            {
                "eval/avg_reward": avg_reward,
                "eval/avg_repetition": avg_repetition,
                "eval/std_repetition": std_repetition,
                "eval/avg_num_decision": avg_decision,
                "eval/avg_eval_time": avg_eval_time,
            },
            step=training_steps,
        )
    print(f"[Evaluation] Training steps: {training_steps} | Average Reward: {avg_reward} | Average Repetition: {avg_repetition} | Std Repetition: {std_repetition}")
    eval_env.close()
    
if __name__ == "__main__":
    main()