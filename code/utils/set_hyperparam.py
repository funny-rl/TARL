from typing import Any
from omegaconf.dictconfig import DictConfig

def set_hyperparam(
    hyper_args: DictConfig, 
    model_args: DictConfig | dict,
    env_info: dict[str, Any],
    device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set hyperparameters based on environment information."""
    
    lr: float = hyper_args.lr
    gamma: float = hyper_args.gamma
    buffer_size: int = hyper_args.buffer_size
    batch_size: int = hyper_args.batch_size
    e_greedy_type: str = hyper_args.e_greedy_type
    max_epsilon: float = hyper_args.max_epsilon
    min_epsilon: float = hyper_args.min_epsilon
    e_decay: int = hyper_args.e_decay
    hidden_dim: int = hyper_args.hidden_dim
    use_image: bool = hyper_args.use_image 
    use_lr_decay: bool = hyper_args.use_lr_decay
    use_hard_update: bool = hyper_args.use_hard_update
    use_act_skip_buf: bool = hyper_args.use_act_skip_buf
    prev_buffer_save: bool = hyper_args.prev_buffer_save
    update_interval: int = hyper_args.update_interval
    tau: float = hyper_args.tau
    algo_name: str = hyper_args.algo_name
    
    if algo_name == "Random":
        base_config = {
            "n_actions": env_info["n_actions"],
            "action_dim": env_info["action_dim"],
            "max_action": env_info.get("max_action", None),
        }
    elif algo_name == "DQN":
        base_config: dict[str, Any] = {
            "state_dim": env_info["state_dim"],
            "action_dim": env_info["action_dim"],
            "n_actions": env_info["n_actions"],
            "lr": lr,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "e_greedy_type": e_greedy_type,
            "e_decay": e_decay,
            "max_epsilon": max_epsilon,
            "min_epsilon": min_epsilon,
            "use_image": use_image,   
            "use_lr_decay": use_lr_decay,
            "use_hard_update": use_hard_update,
            "update_interval": update_interval,
            "tau": tau,
            "use_act_skip_buf": use_act_skip_buf,
            "device": device
        }
    elif algo_name == "DDPG":
        assert use_image == False, "DDPG does not support image input."
        base_config: dict[str, Any] = {
            "state_dim": env_info["state_dim"],
            "action_dim": env_info["action_dim"],
            "max_action": env_info["max_action"],
            "lr": lr,
            "gamma": gamma,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "use_image": use_image,   
            "use_lr_decay": use_lr_decay,
            "use_hard_update": use_hard_update,
            "update_interval": update_interval,
            "tau": tau,
            "expl_noise": hyper_args.expl_noise,
            "use_act_skip_buf": use_act_skip_buf,
            "device": device
        }
    else:
        raise NotImplementedError(f"Algorithm {algo_name} is not supported.")
    
    model_config = {}
    if model_args is not None:
        model_name: str = model_args.model_name
        max_repetition: int = model_args.max_repetition
        rep_batch_size: int = hyper_args.rep_batch_size
        rep_buffer_size: int = hyper_args.rep_buffer_size
        if model_name == "TempoRL":
            model_config = {
                "rep_batch_size": rep_batch_size,
                "rep_buffer_size": rep_buffer_size,
                "max_repetition": max_repetition,
                "e_greedy_type": e_greedy_type,
                "e_decay": e_decay,
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "use_act_skip_buf": use_act_skip_buf,
                "prev_buffer_save": prev_buffer_save,
            }
        elif model_name == "UTE":
            model_config = {
                "rep_batch_size": rep_batch_size,
                "rep_buffer_size": rep_buffer_size,
                "max_repetition": max_repetition,
                "e_greedy_type": e_greedy_type,
                "e_decay": e_decay,
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "num_ensemble": model_args.num_ensemble,
                "use_adaptive_uncertainty": model_args.use_adaptive_uncertainty,
                "uncertainty_factor": model_args.uncertainty_factor,
                "use_act_skip_buf": use_act_skip_buf,
                "prev_buffer_save": prev_buffer_save,
            }
        elif model_name == "EQL":
            assert model_args.coeff_scaling in ["fix", "epsilon", "reverse"], "Invalid coeff_scaling option."
            model_config = {
                "rep_batch_size": rep_batch_size,
                "rep_buffer_size": rep_buffer_size,
                "max_repetition": max_repetition,
                "e_greedy_type": e_greedy_type,
                "e_decay": e_decay,
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "alpha": model_args.alpha,
                "coeff_scaling": model_args.coeff_scaling,
                "sigma": model_args.continuous.sigma,
                "sigma_decay": model_args.continuous.sigma_decay,
                "n_sample": model_args.continuous.n_sample,
                "use_act_skip_buf": use_act_skip_buf,
                "prev_buffer_save": prev_buffer_save,
                "low_variance": model_args.low_variance,
            }
        elif model_name == "TAAC":
            model_config = {
                "temperature": model_args.temperature,
                "target_entropy_delta": model_args.target_entropy_delta,
                "seq_len" : model_args.seq_len,
            }
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        
    return base_config, model_config
