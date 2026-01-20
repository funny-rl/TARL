import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import json
import numpy as np
from repetition_dict import repetition_dict

def calculate_iqm(data_list):
    if not data_list:
        return 0.0
    sorted_data = sorted(data_list)
    n = len(sorted_data)
    trim_count = int(n * 0.25)
    trimmed_data = sorted_data[trim_count : n - trim_count]
    return sum(trimmed_data) / len(trimmed_data)

def get_data(rep_dict, opt_reward, data_path, model_dict): 
    """
    data_path: str, e.g., "data/CliffWalking/DQN"
    model_dist: dict, {
        "EQL" : [list of EQL model names],
        "null" : [list of baseline model names],
        "TempoRL" : [list of TempoRL model names],
        "UTE" : [list of UTE model names]
    }
    """
    data = {}
    for model_type, model_names in model_dict.items():
        data[model_type] = {}
        for model_name in model_names:
            tmp = []
            for seed in range(20):
                opt = True
                best_rep = 0
                visited = 0
                file_path = f"{data_path}/{model_type}/{model_name}/30000/{seed}/eval_log.json"
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
                    

                    tmp.append(best_rep / visited if visited > 0 else 0.0)
            #data[model_type][model_name] = calculate_iqm(tmp)
            data[model_type][model_name] = np.mean(tmp)
    for model_type in data:
        print(f"model type : {model_type}")
        for model_name in data[model_type]:
            print(f"    {model_name} : {data[model_type][model_name]}")
        print()
    return data

def main():
    model_dict = {
        "EQL" : [
            "EQL",
            "EQL_fix",
            "EQL_fix_prev",
            "EQL_prev",
            "EQL_skip",
            "EQL_skip_fix"
        ],

        "null" : [
            "DDQN",
        ],

        "TempoRL" : [
            "TempoRL",
            "TempoRL_prev",
            "TempoRL_skip"
        ],

        "UTE" : [
            "UTE",
            "UTE_prev",
            "UTE_skip"
        ]
    }
    env_names = {"CliffWalking" : -13,
                "Bridge" : -13,
                "ZigZag" : -20}

    for env_name, opt_reward  in env_names.items():
        data_path = "data/"+env_name+"/DQN"
        print(f"=============================Environment: {env_name}=============================")
        data = get_data(repetition_dict[env_name], opt_reward, data_path, model_dict)

if __name__ == "__main__":
    main()