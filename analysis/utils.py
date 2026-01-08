import os
import numpy as np
import pandas as pd
import re


def auto_threshold(data, lambda_factor=5.0):
    # if len(data) == 0:
    #     return None
    # q1 = np.nanpercentile(data, 25)
    # q3 = np.nanpercentile(data, 75)
    # iqr = q3 - q1
    # threshold = q1 - lambda_factor * iqr
    # return threshold
    return np.nanpercentile(data, 5)

def calculate_rl_auc_score(csv_path, output_dir = None, cutoff_percent = 5.0):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return "파일을 찾을 수 없습니다. 경로를 확인해주세요."

    env_name = os.path.basename(csv_path).replace('.csv', '')

    results = []
    step_col = next((c for c in df.columns if c.lower() == 'step'), None)
    if not step_col:
        return "CSV에 'Step' 컬럼이 없습니다."

    reward_cols = [c for c in df.columns if "eval/average_reward" in c and "MIN" not in c and "MAX" not in c]
    all_values = df[reward_cols].values.flatten()

    threshold = auto_threshold(all_values, cutoff_percent)

    min_reward = max(np.nanmin(all_values), threshold)
    max_reward = np.nanmax(all_values)
    print(f"ENV_name : {env_name}")
    print(f"Min Reward: {min_reward}, Max Reward: {max_reward}, threshold: {threshold}")

    for col in reward_cols:
        match = re.search(r"Group:\s*(.*?)\s*-\s*eval/average_reward", col)
        model_name = match.group(1) if match else col
        sub_df = df[[step_col, col]].dropna()
        
        if len(sub_df) < 2:
            continue

        x = sub_df[step_col].values
        y = sub_df[col].values
        y[y < threshold] = threshold


        y_norm = (y - min_reward) / (max_reward - min_reward)
        
        area = np.trapz(y_norm, x)

        total_steps = x[-1] - x[0]
        if total_steps == 0:
            score = 0
        else:
            score = area / total_steps

        results.append({
            'Model': model_name,
            'AUC Score': round(score, 3),
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values(by='AUC Score', ascending=False).reset_index(drop=True)

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{env_name}_auc_scores.csv")
        result_df.to_csv(output_path, index=False)

    return result_df