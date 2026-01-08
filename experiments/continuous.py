import os
import pandas as pd
import numpy as np 

# random score for normalization: 10000 trys
random_reward_dict = {
    "cliffwalking" : -108.9,  
    "zigzag" : -111.5,
    "bridge" : -111.9,
    "fetchreachdense" : -9.876,
    "pendulum" : -1225.555,
    "pointmaze" : 9.652
}

def preprocess(env_name) -> dict:
    data_folder = "./data/main_experiment/"
    df = pd.read_csv(os.path.join(data_folder, f"{env_name}.csv"))
    df = df.drop(columns=df.filter(regex='__MIN|__MAX').columns)

    cols_to_norm = df.columns.difference(['Step'])

    df_norm = df.copy()
    random_reward = random_reward_dict[env_name]
    
    x = df_norm['Step'].values
    algorithms = df_norm.columns.difference(['Step'])
    
    if env_name in ["cliffwalking", "zigzag", "bridge"]:
        denom = df_norm["Group: DDQN_linear - eval/avg_reward"] - random_reward
    else:
        denom = df_norm["Group: DDPG - eval/avg_reward"] - random_reward
    
    df_norm[cols_to_norm] = (
        df_norm[cols_to_norm]
        .sub(random_reward)
        .div(denom, axis=0)
    )
    
    auc_results = {}
    
    for algo in algorithms:
        y = df_norm[algo].values
        area = np.trapezoid(y, x)
        normalized_auc = area / (x.max() - x.min())
        auc_results[algo] = float(round(normalized_auc, 3))
    
    auc_series = pd.Series(auc_results).sort_values(ascending=False)
    
    return auc_series
    
def main():
    
    cliffwalking_auc = preprocess("cliffwalking")
    zigzag_auc = preprocess("zigzag")
    bridge_auc = preprocess("bridge") 
    
    pendulum_auc = preprocess("pendulum")
    fetchreachdense_auc = preprocess("fetchreachdense")
    pointmaze_auc = preprocess("pointmaze")
    
    
    print("[CliffWalking]\n",cliffwalking_auc)
    print("\n[Bridge]\n",bridge_auc)
    print("\n[Zigzag]\n",zigzag_auc)
    
    print("\n-----------------------\n")

    print("\n[Pendulum]\n",pendulum_auc)
    print("[FetchReachDense]\n",fetchreachdense_auc)
    print("\n[PointMaze]\n",pointmaze_auc)

if __name__ == "__main__":
    main()