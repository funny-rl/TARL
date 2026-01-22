import wandb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def fetch_run_data(run):
    run_data = []
    history = run.scan_history(keys=["_step", Project_DIR])
    for row in history:
        if Project_DIR in row:
            run_data.append({
                "run_name": run.name,
                "step": row["_step"],
                Project_DIR: row[Project_DIR]
            })
    return run_data

def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters = {"state" : "finished"})
    
    all_results = []
    
    print(f"Fetching data from {len(runs)} runs using {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(fetch_run_data, runs), total=len(runs), desc="Downloading"))
        
        for result in results:
            all_results.extend(result)

    if all_results:
        df = pd.DataFrame(all_results)
        
        df_pivot = df.pivot_table(
            index='step', 
            columns='run_name', 
            values=Project_DIR, 
        )
                
        df_pivot = df_pivot.sort_index()
        
        print(df_pivot.columns)

        import os
        dir = f"data/{PROJECT}/{METRIC_NAME}"
        os.makedirs(dir, exist_ok=True)
        file_name = f"{dir}/{PROJECT}.csv"
        
        df_pivot.to_csv(file_name)
        
        print(f"\n🚀 변환 성공! 데이터 형태: {df_pivot.shape}")
        print(f"파일 저장 완료: {file_name}")
        
        print("\n--- 데이터 미리보기 ---")
        print(df_pivot.head())

if __name__ == "__main__":
    ENTITY = "singfor7012"
    PROJECT = "CliffWalking"
    METRIC_NAME = "avg_reward"
    Project_DIR = f"eval/{METRIC_NAME}"
    MAX_WORKERS = 8 
    main()