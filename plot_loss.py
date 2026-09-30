"""Plot recorded training loss and its weighted components for one scene."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history_path):
    history_path = Path(history_path)
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    records = [row for row in records if "loss" in row]
    steps = np.array([row["step"] for row in records])
    loss = np.array([row["loss"] for row in records])
    window = min(10, len(records))
    smooth = np.convolve(loss, np.ones(window) / window, mode="valid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(steps, loss, color="#477ca8", alpha=0.35, linewidth=1, label="Recorded total loss")
    axes[0].plot(steps[window - 1:], smooth, color="#16436b", linewidth=1.8,
                 label=f"Mean of {window} recorded samples")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title(history_path.parent.name)
    axes[0].legend()
    # Stage-specific terms contribute zero while that stage is inactive.
    names = dict.fromkeys(name for row in records for name in row['loss_terms'])
    for name in names:
        values = [row["loss_terms"].get(name, 0.) for row in records]
        axes[1].plot(steps, values, linewidth=1, label=name)
    axes[1].set_ylabel("Weighted loss contribution")
    axes[1].set_xlabel("Training step (one sampled training frame per step)")
    axes[1].legend(ncol=3)
    for ax in axes:
        ax.grid(alpha=0.2)
    output = history_path.with_name("loss.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    args = parser.parse_args()
    print(plot_loss(args.history))
