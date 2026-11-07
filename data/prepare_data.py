import argparse
import numpy as np


def normalise(episodes):
    obs = np.concatenate([e["obs"] for e in episodes])
    mean, std = obs.mean(0), obs.std(0) + 1e-6
    for e in episodes:
        e["obs"] = (e["obs"] - mean) / std
    return episodes, mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.parse_args()
