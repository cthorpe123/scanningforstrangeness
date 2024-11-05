import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import random
import time
from matplotlib.colors import ListedColormap, BoundaryNorm

from lib.dataset import ImageDataLoader
from lib.config import ConfigLoader

def visualise_input(input_histogram, height, width):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    img = ax.imshow(input_histogram, origin="lower", cmap='jet', norm=colors.PowerNorm(gamma=0.35, vmin=1e3, vmax=input_histogram.max()), aspect='equal', interpolation='none')
    #cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    #cbar.set_ticks([0., np.max(input_histogram)])
    #cbar.ax.set_yticklabels(['Low charge', 'High charge'], fontsize=12)
    
    ax.set_xticks([0, width - 1])
    ax.set_yticks([0, height - 1])
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xlabel('Wire Coord', fontsize=20)
    ax.set_ylabel('Drift Time', fontsize=20)
    plt.tight_layout()
    return fig

def visualise_truth(target_histogram, height, width):
    cmap = ListedColormap(['#ffffff', '#0000ff', '#ff0000'])  # white, blue, red
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    masked_target_histogram = np.ma.masked_invalid(target_histogram)
    
    img = ax.imshow(masked_target_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')
    
    ax.set_xticks([0, width - 1])
    ax.set_yticks([0, height - 1])
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=18)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)
    ax.set_xlabel('Wire Coord', fontsize=20)
    ax.set_ylabel('Drift Time', fontsize=20)
    
    plt.tight_layout()
    return fig

def visualise(config_file):
    config = ConfigLoader(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_loader = ImageDataLoader(
        input_dir=config.input_dir,
        target_dir=config.target_dir,
        batch_size=1, 
        train_pct=config.train_pct,
        valid_pct=config.valid_pct,
        device=device
    )

    height, width = config.height, config.width
    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    input_img, target_img = next(iter(data_loader.train_dl))

    input_img = input_img.squeeze().cpu().numpy()
    target_img = target_img.squeeze().cpu().numpy()

    unique_values, counts = np.unique(target_img, return_counts=True)
    print("Unique Values and Counts in Truth Histogram:")
    for value, count in zip(unique_values, counts):
        print(f"Value {value}: {count} occurrences")

    timestamp = int(time.time())

    input_fig = visualise_input(input_img, height, width)
    input_path = os.path.join(plot_dir, f"input_event_{timestamp}.png")
    input_fig.savefig(input_path, dpi=300)
    plt.close(input_fig)

    target_fig = visualise_truth(target_img, height, width)
    target_path = os.path.join(plot_dir, f"target_event_{timestamp}.png")
    target_fig.savefig(target_path, dpi=300)
    plt.close(target_fig)

    print(f"Saved input histogram to {input_path}")
    print(f"Saved target histogram to {target_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    
    visualise(args.config)
