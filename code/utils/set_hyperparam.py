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
    update_interval: int = hyper_args.update_interval
    tau: float = hyper_args.tau
    algo_name: str = hyper_args.algo_name
    max_grad_norm: float = hyper_args.max_grad_norm
    use_ddqn: bool = hyper_args.use_ddqn
    use_dueling: bool = hyper_args.use_dueling
    
    if algo_name == "DQN":
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
            "max_grad_norm": max_grad_norm,
            "use_ddqn": use_ddqn,
            "use_dueling": use_dueling,
            "device": device
        }
    else:
        raise NotImplementedError(f"Algorithm {algo_name} is not supported.")
    
    model_config = {}
    if model_args is not None:
        model_name: str = model_args.model_name
        max_repetition: int = model_args.max_repetition
        if model_name == "TempoRL":
            model_config = {
                "max_repetition": max_repetition,
                "e_greedy_type": e_greedy_type,
                "e_decay": e_decay,
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "use_dueling": use_dueling,
            }
        elif model_name == "UTE":
            model_config = {
                "max_repetition": max_repetition,
                "e_greedy_type": e_greedy_type,
                "e_decay": e_decay,
                "max_epsilon": max_epsilon,
                "min_epsilon": min_epsilon,
                "use_dueling": use_dueling,
                "num_ensemble": model_args.num_ensemble,
                "uncertainty_factor": model_args.uncertainty_factor,
            }
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")
        
    return base_config, model_config