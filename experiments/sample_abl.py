import os
import pandas as pd
import numpy as np 

from rliable import library as rly
from rliable import metrics



def preprocess(env_name, algo_name) -> dict:
    """
    Each data: csv files with columns: step, seeds (10 seeds)
    """
    
    data_folder = os.path.join(f"./data/main_exp/{env_name}/{algo_name}.csv")
    df = pd.read_csv(data_folder)
    
    # remove the __MAX, __MIN columns
    df = df[[col for col in df.columns if "__MAX" not in col and "__MIN" not in col]] 
    
    final_scores = []
    steps = df['Step'].values
    max_step = steps[-1]
    min_step = steps[0]
    total_range = max_step - min_step
    
    for col in df.columns:
        if col != "Step":
            seed_values = df[col].values
            # AUC normalization
            normalized_values = (seed_values - random_reward_dict[env_name]) / (max_score_dict[env_name] - random_reward_dict[env_name])
            # 면적 계산
            auc = np.trapezoid(normalized_values, x=steps) / total_range
            final_scores.append(auc)
    np_scores = np.array([final_scores]).T  # shape: (num_seeds, num_steps)
    
    return np_scores

def main():
    # envs = ["cliff",  "bridge", "zigzag"]
    # algos = [
    #     "DDPG", 
    #     "TempoRL", 
    #     "TempoRL_skip", 
    #     "TempoRL_prev",
    #     "UTE_prev",
    #     "UTE_skip",
    #     "UTE",
    #     "EQL", 
    #     "EQL_fix",
    #     "EQL_skip",
    #     "EQL_skip_fix",
    #     "EQL_prev",
    #     "EQL_fix_prev",
    # ]
    envs = [
        "pendulum", 
        "fetchreachdense",
        "pointmaze"
    ]
    algos = [
        "DDPG", 
        "TempoRL", 
        "TempoRL_skip", 
        "UTE",
        "UTE_skip",
        "EQL", 
        "EQL_fix",
        "EQL_skip",
        "EQL_skip_fix",
        "TAAC"
    ]
    
    print("=" * 100)
    print(f"{'ENV':<10} | {'ALGORITHM RANKING (Descending by IQM Score)':<90}")
    print("-" * 100)

    for env in envs:
        algo_scores = {}
        for algo in algos:
            np_scores = preprocess(env, algo)
            algo_scores[algo] = np_scores

        aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x)])
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            algo_scores, aggregate_func, reps=2000
        )

        sorted_algos = sorted(
            iqm_scores.keys(), 
            key=lambda x: iqm_scores[x][0], 
            reverse=True
        )

        row_str = f"{env:<10} | "
        for rank, algo in enumerate(sorted_algos, 1):
            score = iqm_scores[algo][0]
            low = iqm_cis[algo][0][0]
            high = iqm_cis[algo][1][0]
            
            cell = f"[{rank}] {algo}: {score:.3f}({low:.3f}-{high:.3f})"
            row_str += f"{cell}  >>  "
        
        print(row_str.rstrip(" >> "))
        print("-" * 100)

if __name__ == "__main__":
    main()
    