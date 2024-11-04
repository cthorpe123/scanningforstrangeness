import os
import numpy as np
import matplotlib.pyplot as plt
import torch 

from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from lib.dataset import ImageDataLoader
from lib.config import ConfigLoader

def visualise_input(input_histogram):
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(input_histogram, origin="lower", cmap='jet', norm=LogNorm(vmin=1e3), aspect='equal', interpolation='none')
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Charge")

    ax.set_xticks([0, 511])
    ax.set_yticks([511])
    ax.tick_params(axis="y", which='major', direction="out", length=10, width=2.5, pad=10, labelsize=50)
    ax.tick_params(axis="y", which='minor', direction="out", length=10, width=1.0, labelleft=False, labelsize=50)
    ax.tick_params(axis="x", which='major', direction="out", length=10, width=2.5, pad=10, bottom=True, top=False, labelsize=50)
    ax.tick_params(axis="x", which='minor', direction="out", length=10, width=2.0, bottom=True, top=False, labelsize=50)
    ax.set_xlim(0, 511)
    ax.set_ylim(0, 511)
    ax.set_xlabel('Wire Coord', size=55, labelpad=1.0)
    ax.set_ylabel('Drift Time', size=55, labelpad=1.0)
    
    plt.tight_layout()
    return fig


def visualise_truth(target_histogram):
    cmap = ListedColormap(['white', 'red', 'blue', 'cyan', 'green', 'yellow', 'purple'])
    bounds = np.arange(0, target_histogram.max() + 2)
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    masked_target_histogram = np.ma.masked_invalid(target_histogram)  
    img = ax.imshow(masked_target_histogram, cmap=cmap, norm=norm, aspect='equal', interpolation='none')
    
    ax.set_xticks([0, 511])
    ax.set_yticks([0, 511])
    ax.tick_params(axis="y", which='major', direction="out", length=10, width=2.5, pad=10, labelsize=50)
    ax.tick_params(axis="y", which='minor', direction="out", length=10, width=1.0, labelleft=False, labelsize=50)
    ax.tick_params(axis="x", which='major', direction="out", length=10, width=2.5, pad=10, bottom=True, top=False, labelsize=50)
    ax.tick_params(axis="x", which='minor', direction="out", length=10, width=2.0, bottom=True, top=False, labelsize=50)
    ax.set_xlim(0, 511)
    ax.set_ylim(0, 511)
    ax.set_xlabel('Wire Number', size=55, labelpad=1.0)
    ax.set_ylabel('Drift Time', size=55, labelpad=1.0)
    
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

    os.makedirs(os.path.join(config.output_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "target"), exist_ok=True)

    for i, (input_img, target_img) in enumerate(data_loader.train_dl):
        input_fig = visualise_input(input_img[0].cpu().numpy())
        input_path = os.path.join(config.output_dir, "input", f"input_event_{i}.png")
        input_fig.savefig(input_path, dpi=300)
        plt.close(input_fig)

        target_fig = visualise_truth(target_img[0].cpu().numpy())
        target_path = os.path.join(config.output_dir, "target", f"target_event_{i}.png")
        target_fig.savefig(target_path, dpi=300)
        plt.close(target_fig)

        break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    visualise(args.config)
