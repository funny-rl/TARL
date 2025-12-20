import argparse

from random_grid import random_grid
from grid_env import state_visitation_heatmap

if __name__ == "__main__":
    Fig_Registry = {
        "random_grid": random_grid,
        "state_visitation_heatmap": state_visitation_heatmap,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure_name", 
        type=str, 
        default="random_grid", 
        help="Name of the figure to generate"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CliffWalking",
        help="Name of the environment"
    )

    args = parser.parse_args()
    Fig_Registry[args.figure_name](args)

