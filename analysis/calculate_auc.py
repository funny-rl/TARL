import argparse

from utils import calculate_rl_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Directory to save the AUC scores CSV"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save the AUC scores CSV"
    )
    parser.add_argument(
        "--cutoff_percent",
        type=float,
        default=5.0,
        help="Threshold for reward normalization"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing RL evaluation results"
    )

    args = parser.parse_args()
    path = args.data_dir+args.csv_path
    result_df = calculate_rl_auc_score(
        path,
        output_dir=args.output_dir,
        cutoff_percent=args.cutoff_percent
    )
    print(result_df)

