# scripts/analyze_logs.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="logs", help="folder with episode_XXXX.csv files")
    args = ap.parse_args()

    files = sorted(Path(args.logdir).glob("episode_*.csv"))
    if not files:
        print("No CSVs found.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["episode_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"Skip {f}: {e}")

    data = pd.concat(dfs, ignore_index=True)
    # Example: plot per-turn reward distribution
    plt.figure()
    data["reward"].hist(bins=50)
    plt.title("Per-turn Reward Distribution")
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.show()

if __name__ == "__main__":
    main()
