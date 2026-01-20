import wandb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def fetch_run_data(run):
    """단일 Run에서 데이터를 가져오는 함수"""
    run_data = []
    # 필요한 key만 골라서 가져오기
    history = run.scan_history(keys=["_step", METRIC_NAME])
    for row in history:
        if METRIC_NAME in row:
            run_data.append({
                "run_name": run.name,
                "step": row["_step"],
                METRIC_NAME: row[METRIC_NAME]
            })
    return run_data

def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters = {"state" : "finished"})
    
    all_results = []
    
    print(f"Fetching data from {len(runs)} runs using {MAX_WORKERS} threads...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map의 결과를 리스트로 변환
        results = list(tqdm(executor.map(fetch_run_data, runs), total=len(runs), desc="Downloading"))
        
        for result in results:
            all_results.extend(result)

    if all_results:
        df = pd.DataFrame(all_results)
        
        df_pivot = df.pivot_table(
            index='step', 
            columns='run_name', 
            values=METRIC_NAME, 
            aggfunc='mean'  # 같은 step에 값이 여러 개라면 평균값을 사용
        )
                
        df_pivot = df_pivot.sort_index()

        import os
        os.makedirs("wandb_data", exist_ok=True)
        file_name = f"wandb_data/{PROJECT}_eval_reward_pivoted.csv"
        
        df_pivot.to_csv(file_name)
        
        print(f"\n🚀 변환 성공! 데이터 형태: {df_pivot.shape}")
        print(f"파일 저장 완료: {file_name}")
        
        # 샘플 출력
        print("\n--- 데이터 미리보기 ---")
        print(df_pivot.head())

if __name__ == "__main__":
    ENTITY = "singfor7012"
    PROJECT = "ZigZag"
    METRIC_NAME = "eval/avg_reward"
    MAX_WORKERS = 8 
    main()