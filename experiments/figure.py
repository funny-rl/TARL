import os
import json 
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly

from matplotlib.ticker import FuncFormatter

from rliable import metrics
from scipy.stats import trim_mean
from utils.dicts import ENV_DICT
from utils.repetition_dict import get_optimal_repetitions
from utils.dicts import random_reward_dict, max_score_dict

color_map = {
    "DDQN": "#F11FDC",
    "DDPG": "#F11FDC",
    "TempoRL": "#1f77b4",
    "TempoRL-M": "#1f77b4",
    "UTE": "#D83A16",
    "UTE-M": "#D83A16",
    "RARe(ours)": "#0c0c0b",
    "RARe-M(ours)": "#0c0c0b",
    "TAAC": "#bcbd22",
}

line_styles = {
    "DDQN": "-",
    "DDPG": "-",
    "TempoRL": "-",
    "UTE": "-",
    "RARe(ours)": "-",
    "UTE-M": "--",
    "TempoRL-M": "--",
    "RARe-M(ours)": "--",
    "TAAC": "-",
    
}

def format_x(x, pos):
    return f"{x/10000:.1f}"

def smooth(values, window=5):
    if window <= 1:
        return values

    pad_size = window // 2
    padded_values = np.pad(values, (pad_size, pad_size), mode='edge')
    
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded_values, kernel, mode='valid')
    
    return smoothed[:len(values)]


def get_reward_data(env_name, csv_file):
    df = pd.read_csv(csv_file)
    final_scores = []
    steps = df['step'].values
    for col in df.columns:
        if col != "step":
            seed_values = df[col].values
            normalized_values = (seed_values - random_reward_dict[env_name]) / (max_score_dict[env_name] - random_reward_dict[env_name])
            final_scores.append(normalized_values)
    return np.array(final_scores), steps
    

def get_repetition_data(env_name, data_path):
    data = []
    env_config = ENV_DICT[env_name]
    rep_dict = get_optimal_repetitions(env_config["pits_list"], env_config["shape"], env_config["goal"], MAX_Repetition)
    for step in range(0, 20001, 200):
        tmp = []
        for seed in range(20):
            best_rep = 0
            visited = 0
            file_path = f"{data_path}/{step}/{seed}/eval_log.json"
            prev_action, prev_rep = -1, -1
            with open(file_path, "r") as f:
                logs = json.load(f)
                for log in logs:
                    if "state" not in log:
                        continue
                    state_vec = log["state"][0]
                    state_idx = int(np.argmax(state_vec))
                    if state_idx in rep_dict:
                        options = rep_dict[state_idx]
                        visited += 1
                        if log["action"][0] in options:
                            if log["repetition"] in options[log["action"][0]]:
                                best_rep += 1
                                if log["action"][0] == prev_action and max(log["repetition"], prev_rep) != MAX_Repetition:
                                    best_rep -= 1
                            prev_action = log["action"][0]
                            prev_rep = log["repetition"]
                tmp.append(best_rep/visited)
        # IQM score and CI for each step, over seeds
        data.append(tmp)   
    return np.array(data).T, np.arange(0, 20001, 200) # (Seeds, Steps), (Steps, )

def preprocess_for_curve(env_name, algo_info, is_repetition_mode):
    algo_key, algo_name = algo_info
    try:
        if is_repetition_mode:
            data = {}
            data_path = f"./data/{env_name}/DQN/{algo_key}/{algo_name}"
            data, steps = get_repetition_data(env_name, data_path)
            
                    
        else:
            csv_file = f"./data/{env_name}/avg_reward/{algo_name}.csv"
            data, steps = get_reward_data(env_name, csv_file)

    except Exception as e:
        print(f"Error occurred while processing {env_name} - {algo_name}: {e}")
        return None, None
    
    return data, steps


def main():
    if DISCRETE:
        from utils.dicts import D_MODEL_DICT as MODEL_DICT

    else:
        pass
    envs = list(MODEL_DICT.keys())
    num_envs = len(MODEL_DICT.keys())
    fig, axes = plt.subplots(
        1, num_envs, 
        figsize=(5*num_envs, 6), 
        sharex=False, 
        sharey=False
    )
    
    all_lines = []
    algo_labels = []
    
    for ax, env in zip(axes, MODEL_DICT.keys()):
        algos = []
        models = []
        for algo_key, algo_list in MODEL_DICT[env].items():
            for algo_dict in algo_list:
                for model_name, algo_name in algo_dict.items():
                    models.append(model_name)
                    algos.append([algo_key, algo_name])
        
        assert len(algos) == len(models ), "Mismatch between algos and models lengths"
                        
        for algo, model in tqdm(zip(algos, models), desc=f"Processing {env}", total=len(models)):
            prefix_name  = algo[1]
            data, steps = preprocess_for_curve(env, algo, Repetition)
            if data is None:
                print(f"Data not found for {env} - {prefix_name}")
                continue
            aggregate_func = lambda x: trim_mean(x, proportiontocut=0.25, axis=0)
            
            scores, cis = rly.get_interval_estimates(
                {model: data}, 
                aggregate_func, 
                reps=2000
            )

            iqm_values = scores[model]
            low_ci = cis[model][0]
            high_ci = cis[model][1]    
            
            iqm_values = smooth(iqm_values)
            low_ci = smooth(low_ci)
            high_ci = smooth(high_ci)

            line, = ax.plot(
                steps, 
                iqm_values, 
                label=model, 
                linewidth=2.5,
                color=color_map[model],
                linestyle=line_styles[model],
            )
            ax.fill_between(steps, low_ci, high_ci, color=line.get_color(), alpha=0.15)

            if env == envs[-1]:
                all_lines.append(line)
                algo_labels.append(model)
        
        if env == "ZigZag":
            ax.set_xlim(0, 12500)
            ax.set_ylim(0.05, 1.05)
            ax.set_xticks(np.arange(0, 12501, 2500))
            
        
        elif env == "Bridge":
            ax.set_xlim(0, 8000)
            ax.set_ylim(0.05, 1.05)
            ax.set_xticks(np.arange(0, 8001, 2000))
            
        elif env == "CliffWalking":
            ax.set_xlim(0, 7500)
            ax.set_ylim(0.05, 1.05)
            ax.set_xticks(np.arange(0, 7501, 2500))


        elif env == "pendulum":
            ax.set_xlim(0, 5500)
            ax.set_ylim(-0.01, 1.1)
            ax.set_xticks(np.arange(0, 5501, 1000))
        
        elif env == "fetchreachdense":
            ax.set_xlim(0, 50000)
            ax.set_ylim(-0.01, 1.1)
            ax.set_xticks(np.arange(0, 50001, 10000))
            
        elif env == "pointmaze":
            ax.set_xlim(0, 250000)
            ax.set_ylim(-0.01, 2.4)
            ax.set_xticks(np.arange(0, 250001, 50000))

        ax.tick_params(axis='both', labelsize=16) 
        ax.set_title(env.upper(), fontsize=16, fontweight='bold', pad=12) 
        ax.xaxis.set_major_formatter(FuncFormatter(format_x))
        ax.grid(True, linestyle='-', alpha=0.8)

    if DISCRETE:
        fig.legend(
            all_lines, 
            algo_labels, 
            loc='upper center', 
            ncol=len(algos), 
            fontsize=16,
        )
    else:    
        fig.legend(
            all_lines, 
            algo_labels, 
            loc='upper center', 
            ncol=len(algos), 
            fontsize=16
        )

    fig.supxlabel("Training Steps ($\\times 10^4$)", fontsize=16)
    fig.supylabel("IQM Normalized Score", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.92]) 

    os.makedirs("./figures", exist_ok=True)
    fig_name1 = "discrete" if DISCRETE else "continuous"
    fig_name2 = "repetition" if Repetition else "reward"
    fig.savefig(f"./figures/{fig_name1}_{fig_name2}_iqm_curve.pdf", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    MAX_Repetition = 4
    DISCRETE = True
    Repetition = False
    main()