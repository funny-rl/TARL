import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import argparse



def random_grid(args):
    folder_path = "./data/random_walk"
    files = [
        "random_grid_uniform_repetition1_center_grid75_num_experiments10.csv",
        "random_grid_geometric_repetition20_center_grid75_num_experiments10.csv",
        "random_grid_uniform_repetition20_center_grid75_num_experiments10.csv"
    ]
    plt.figure(figsize=(10, 6))

    for file in files:
        df = pd.read_csv(f"{folder_path}/{file}")
        x = np.arange(1, len(df.columns) + 1)
        mean_vals = df.mean(axis=0)
        std_vals = df.std(axis=0)
        line, = plt.plot(x, mean_vals, label=f"{file}", linewidth=2)
        plt.fill_between(
            x,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=line.get_color(),
            alpha=0.2
        )
    plt.xlim(1, 50)
    plt.title("Mean and Std Dev of Visitation Log-Probabilities", fontsize=14)
    plt.xlabel("Search Range (Distance from Center)", fontsize=12)
    plt.ylabel("Log Probability", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = "./figure/random_grid_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")
    plt.show()

actions = [
    (0, 1), # Up
    (1, 0), # Right 
    (-1, 0), # Left
    (0, -1) # Down
]  

def main():
    
    datas: list[list[float]] = []
    for seed in range(num_experiments):
        seed += 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        visit_count = defaultdict(int)
        for _ in tqdm(range(N_episodes), desc="Simulating Episodes"):
            pos = (CENTER_GRID, CENTER_GRID)
            done = False
            step = 0
            while not done:
                visit_count[pos] += 1
                action = actions[np.random.choice(len(actions))]
                if rep_sampling == "uniform":
                    repetition = np.random.randint(1, Max_Repetition + 1)
                elif rep_sampling == "geometric":
                    # truncated geometric distribution
                    rep_p = torch.tensor(0.3)  # parameter for geometric distribution
                    u = torch.rand(1)
                    trunc_cdf_max = 1 - (1 - rep_p) ** Max_Repetition
                    u_prime = u * trunc_cdf_max
                    k = torch.ceil(torch.log(1 - u_prime) / torch.log(1 - rep_p))
                    k = torch.clamp(k, min=1, max=Max_Repetition)
                    repetition = int(k.item())
                else:
                    raise ValueError("Invalid repetition sampling method.")
                for _ in range(repetition):
                    step += 1
                    new_pos = (pos[0] + action[0], pos[1] + action[1])
                    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
                        pos = new_pos

                    if step == episode_length:
                        done = True
                        break
            
        print(f"Num visited states: {len(visit_count.keys())} / {GRID_SIZE * GRID_SIZE}")
        prob = {s: c / N_episodes for s, c in visit_count.items()}
        prob_grid = np.zeros((GRID_SIZE, GRID_SIZE))

        for (x, y), p in prob.items():
            prob_grid[x, y] = p
        
        prob_grid += 1e-10 
        prob_grid = np.log(prob_grid)
        data: float[float] = []
        for search_range in range(1, CENTER_GRID + 1):
            neighbor_log_prob = np.mean(
                prob_grid[CENTER_GRID - search_range:CENTER_GRID + search_range,CENTER_GRID - search_range: CENTER_GRID + search_range]
            )
            data.append(round(neighbor_log_prob.item(), 3))

        print(data)
        datas.append(data)
    df = pd.DataFrame(datas)
    df.to_csv(
        os.path.join(
            output_dir,
            f"./random_grid_{rep_sampling}_repetition{Max_Repetition}_center_grid{CENTER_GRID}_num_experiments{num_experiments}.csv"
        ),
        index=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(
        prob_grid,
        cmap="gray",
        interpolation="nearest"
    )
    plt.colorbar(label="Visitation Probability")

    plt.savefig(
        f"./image/visitation_heatmap_{rep_sampling}_repetition{Max_Repetition}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_repetition", type=int, default=20)
    parser.add_argument(
        "--rep_sampling",
        type=str,
        choices=["uniform", "geometric"],
        default="uniform"
    )
    parser.add_argument("--N_episodes", type=int, default=20000)
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument("--center_grid", type=int, default=75)
    args = parser.parse_args()

    output_dir = "./data"
    num_experiments = 10
    Max_Repetition = args.max_repetition
    rep_sampling = args.rep_sampling
    N_episodes = args.N_episodes
    episode_length = args.episode_length

    CENTER_GRID = args.center_grid
    GRID_SIZE = 2 *  CENTER_GRID

    main()