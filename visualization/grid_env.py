import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

env_info = {
    "CliffWalking": {
        "shape": [4, 12],
        "start": [3, 0],
        "goal": [3, 11],
        "pits": [
            [3,1], [3,2], [3,3], [3,4], [3,5],
            [3,6], [3,7], [3,8], [3,9], [3,10],
        ]
    },
    "Bridge": {
        "shape": [6, 10],
        "start": [0, 0],
        "goal": [0, 9],
        "pits": [
            [0,2], [1,2], [0,3], [1,3], [0,4],
            [1,4], [0,5], [1,5], [0,6], [1,6],
            [0,7], [1,7], [4,2], [5,2], [4,3],
            [5,3], [4,4], [5,4], [4,5], [5,5],
            [4,6], [5,6], [4,7], [5,7],
        ]
    },
    "ZigZag": {
        "shape": [6, 10],
        "start": [0, 0],
        "goal": [5, 9],
        "pits": [
            [0,2], [1,2], [2,2], [3,2], [0,3],
            [1,3], [2,3], [3,3], [2,6], [3,6],
            [4,6], [5,6], [2,7], [3,7], [4,7],
            [5,7],
        ]
    }
}

def state_visitation_heatmap(args):
    env_name = args.env_name
    folder_path = f"../code/videos/{env_name}/DQN/"
    max_step = 100000
    model_path = [
        f"null/DQN/{max_step}/",
        f"TempoRL/TempoRL/{max_step}/",
        f"UTE/UTE_-2/{max_step}/",
        f"UTE/UTE_2/{max_step}/",
        f"EQL/EQL_uniform/{max_step}/",
        f"EQL/EQL_geo03/{max_step}/",
    ]

    for model in model_path:
        experiment_path = os.path.join(folder_path, model)
        sub_folders = glob.glob(os.path.join(experiment_path, "*/"))

        if not sub_folders:
            raise ValueError(f"No sub-folders found in {experiment_path}")

        num_seeds = len(sub_folders)
        state_map = defaultdict(int)

        for sub_folder in sub_folders:
            tracking_log = os.path.join(sub_folder, "eval_log.json")
            if not os.path.exists(tracking_log):
                continue
                
            with open(tracking_log, "r") as f:
                data = json.load(f)
            
            for record in data[1:-1]:
                state = record["state"][0]
                state_idx = int(np.argmax(state))
                state_map[state_idx] += 1 
        
        shape = env_info[env_name]["shape"]
        heatmap = np.zeros(shape)
        for idx, count in state_map.items():
            r, c = np.unravel_index(idx, shape)
            heatmap[r, c] = count / num_seeds

        plt.figure(figsize=(12, 5))
        im = plt.imshow(heatmap, cmap='YlOrBr', interpolation='nearest')
        plt.colorbar(im, label='Avg. Visitation Count')

        start = env_info[env_name]["start"]
        goal = env_info[env_name]["goal"]
        pits = env_info[env_name]["pits"]

        ax = plt.gca()
        for r in range(shape[0]):
            for c in range(shape[1]):
                rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.3)
                ax.add_patch(rect_border)
                
                if [r, c] == start:
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='green', alpha=0.4)
                    ax.add_patch(rect)
                    plt.text(c, r, 'S', ha='center', va='center', fontweight='bold', color='darkgreen')
                
                elif [r, c] == goal:
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='blue', alpha=0.4)
                    ax.add_patch(rect)
                    plt.text(c, r, 'G', ha='center', va='center', fontweight='bold', color='darkblue')
                
                elif [r, c] in pits:
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='red', alpha=0.4)
                    ax.add_patch(rect)

        plt.title(f"State Visitation Heatmap: {model.split('/')[0]}")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")

        save_dir = f"./figure/state_visitation_heatmap/{env_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        img_save_path = os.path.join(save_dir, f"{model.replace('/', '_')}_heatmap.png")
        plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {img_save_path}")