import os
import re
import wandb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_base_name(run_name):
    base = re.sub(r'(DQN_|DDPG_)', '', run_name)
    base = re.sub(r'(None_|TempoRL_|UTE_|EQL_)', '', base)
    base = re.sub(r'(CliffWalking_|Bridge_|ZigZag_)', '', base)
    return base

def fetch_run_data(run):
    run_data = []
    history = run.scan_history(keys=["_step", Project_DIR])
    
    if not run.group:
        raise ValueError(f"Run {run.name} does not belong to any group.")
    else:
        base_name = get_base_name(run.name)
        group_name = run.group if run.group else "no_group"
        
    for row in history:
        if Project_DIR in row:
            run_data.append({
                "group": group_name,
                "run_name": base_name,
                "step": row["_step"],
                Project_DIR: row[Project_DIR]
            })
    return run_data

def main():
    api = wandb.Api(timeout=60)
    for PROJECT in PROJECTS:
        runs = api.runs(f"{ENTITY}/{PROJECT}", filters = {"state" : "finished"})
        
        all_results = []
        
        print(f"Fetching data from {len(runs)} runs using {MAX_WORKERS} threads...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(fetch_run_data, runs), total=len(runs), desc="Downloading"))
            
            for result in results:
                all_results.extend(result)

        if all_results:
            full_df = pd.DataFrame(all_results)
            print(f"{PROJECT}")
            for group_name, group_df in full_df.groupby('group'):
                        
                        df_pivot = group_df.pivot_table(
                            index='step', 
                            columns='run_name', 
                            values=Project_DIR
                        ).sort_index()

                        output_dir = f"data/{PROJECT}/{METRIC_NAME}"
                        os.makedirs(output_dir, exist_ok=True)
                        file_name = f"{output_dir}/{group_name}.csv"
                        
                        df_pivot.to_csv(file_name)

if __name__ == "__main__":
    ENTITY = "singfor7012"
    #["CliffWalking", "Bridge", "ZigZag"]
    #["Pendulum-v1", "LunarLander-v3"]
    #["PointMaze_Large_Diverse_GDense-v3"]
    PROJECTS = ["PointMaze_Large_Diverse_GDense-v3"] 
    METRIC_NAME = "avg_reward"
    Project_DIR = f"eval/{METRIC_NAME}"
    MAX_WORKERS = 16
    main()