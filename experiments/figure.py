import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly

from matplotlib.ticker import FuncFormatter

from rliable import metrics
from scipy.stats import trim_mean
from dicts import random_reward_dict, max_score_dict

color_map = {
    "DDQN": "#F11FDC",
    "DDPG": "#F11FDC",
    "TempoRL": "#1f77b4",
    "TempoRL_skip": "#1f77b4",
    "UTE": "#D83A16",
    "UTE_skip": "#D83A16",
    "EQL": "#0c0c0b",  
    "EQL_skip": "#0c0c0b",
    "TAAC": "#bcbd22",
}

line_styles = {
    "DDQN": "-",
    "DDPG": "-",
    "TempoRL": "-",
    "TempoRL_skip": "--",
    "UTE": "-",
    "EQL": "-",
    "UTE_skip": "--",
    "EQL_skip": "--",
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

def preprocess_for_curve(env_name, algo_name):
    try:
        data_path = f"./data/main_exp/{env_name}/{algo_name}.csv"
        df = pd.read_csv(data_path)
    except:
        return None, None
    
    reward_cols = [col for col in df.columns if col != "Step" and "__" not in col]
    steps = df['Step'].values
    raw_rewards = df[reward_cols].values # (Steps, Seeds)
    
    raw_rewards = np.clip(raw_rewards, random_reward_dict[env_name], None)
    
    # 정규화
    rand_r = random_reward_dict[env_name]
    max_r = max_score_dict[env_name]
    normalized_rewards = (raw_rewards - rand_r) / (max_r - rand_r)
    
    # rliable 계산을 위해 (Seeds, Steps)로 변환
    return normalized_rewards.T, steps

def main():
    if DISCRETE:
        envs = ["cliff", "bridge", "zigzag"]
        algos = [
            "DDQN",
            "TempoRL",
            "UTE",
            "EQL",
            "TempoRL_skip",
            "UTE_skip",
            "EQL_skip",
        ]
    else:
        envs = ["pendulum", "fetchreachdense", "pointmaze"] 
        algos = [
            "DDPG", 
            "TempoRL",
            "UTE", 
            "TAAC", 
            "EQL",
            "UTE_skip",
            "EQL_skip",
            "TempoRL_skip",
        ] 
        
    fig, axes = plt.subplots(
        1, len(envs), 
        figsize=(5*len(envs), 6), 
        sharex=False, 
        sharey=False
    )

    all_lines = []
    algo_labels = []
    
    for ax, env in zip(axes, envs):
        
        for algo in algos:
            data, steps = preprocess_for_curve(env, algo)
            if data is None: continue

            aggregate_func = lambda x: trim_mean(x, proportiontocut=0.25, axis=0)
            
            # {algo: data} 형태로 전달
            scores, cis = rly.get_interval_estimates(
                {algo: data}, 
                aggregate_func, 
                reps=2000
            )
            
            iqm_values = scores[algo]   
            low_ci = cis[algo][0]
            high_ci = cis[algo][1]    
            
            iqm_values = smooth(iqm_values)
            low_ci = smooth(low_ci)
            high_ci = smooth(high_ci)

            line, = ax.plot(
                steps, 
                iqm_values, 
                label=algo, 
                linewidth=2.5,
                color=color_map.get(algo),
                linestyle=line_styles.get(algo)
            )
            ax.fill_between(steps, low_ci, high_ci, color=line.get_color(), alpha=0.15)

            if env == envs[-1]:
                all_lines.append(line)
                algo_labels.append(algo)
        
        if env == "zigzag":
            ax.set_xlim(0, 12500)
            ax.set_ylim(0.05, 1.05)
            ax.set_xticks(np.arange(0, 12501, 2500))
        
        elif env == "bridge":
            ax.set_xlim(0, 12500)
            ax.set_ylim(0.05, 1.05)
            ax.set_xticks(np.arange(0, 12501, 2500))
            
        elif env == "cliff":
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

        ax.tick_params(axis='both', labelsize=16) # 축 숫자 크기
        ax.set_title(env.upper(), fontsize=16, fontweight='bold', pad=12) # 환경 이름 크기
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

    # 공통 축 라벨
    fig.supxlabel("Training Steps ($\\times 10^4$)", fontsize=16)
    fig.supylabel("IQM Normalized Score", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.92])  # legend 공간 확보

    os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"./figures/{fig_name}_iqm_curve.pdf", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    DISCRETE = True
    if DISCRETE:
        fig_name = "discrete"
    else:
        fig_name = "continuous"
    main()