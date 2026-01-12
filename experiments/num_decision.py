import os
import pandas as pd
import numpy as np 

from rliable import library as rly
from rliable import metrics

from dicts import decision_dict

def preprocess(env_name, algo_name) -> dict:
    """
    Each data: csv files with columns: step, seeds (10 seeds)
    """
    

 
    
    final_scores = []
    
    for decisions in decision_dict[env_name][algo_name]:
        
        final_scores.append(decisions)
    np_scores = np.array([final_scores]).T  # shape: (num_seeds, num_steps)
    
    return np_scores

def main():
    envs = [
        "cliff",  
        "bridge", 
        "zigzag",
        "pendulum",
        "fetchreachdense",
        "pointmaze"
    ]
    algos = [
        "TempoRL", 
        "TempoRL_skip", 
        "UTE_skip",
        "UTE",
        "EQL", 
        "EQL_fix",
        "EQL_skip",
        "EQL_skip_fix",
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
    