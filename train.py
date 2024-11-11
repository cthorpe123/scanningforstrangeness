import torch.nn as nn
import time
from line_profiler import profile
from torch.cuda.amp import autocast, GradScaler

import os
import torch
import numpy as np
import uproot
import argparse
from datetime import datetime

from lib.dataset import ImageDataLoader
from lib.model import UNet
from lib.loss import FocalLoss
from lib.common import set_seed
from lib.config import ConfigLoader

def create_model(n_classes, weights, device):
    model = UNet(1, n_classes=n_classes, depth=4, n_filters=16)
    #loss_fn = FocalLoss(alpha=weights, gamma=2.0) 
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, loss_fn, optim

def get_class_weights(stats):
    if np.any(stats == 0.):
        idx = np.where(stats == 0.)
        stats[idx] = 1
        weights = 1. / stats
        weights[idx] = 0
    else:
        weights = 1. / stats
    return [weight / sum(weights) for weight in weights]

def calculate_or_load_class_counts(data_loader, num_classes, cache_path="class_counts.npy"):
    if os.path.exists(cache_path):
        print("\033[94m-- Loading cached class counts\033[0m")
        return np.load(cache_path)
    else:
        print("\033[94m-- Counting classes from scratch\033[0m")
        class_counts = data_loader.count_classes(num_classes)
        np.save(cache_path, class_counts)
        return class_counts

def train(config):
    print("\033[94m-- Initialising training configuration\033[0m")
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    print(f"\033[94m-- Using device: {device}\033[0m")

    batch_size = config.batch_size
    train_pct = config.train_pct
    valid_pct = config.valid_pct
    n_classes = config.n_classes
    n_epochs = config.n_epochs

    input_dir = config.input_dir
    target_dir = config.target_dir
    output_dir = config.output_dir
    model_name = config.model_name

    print("\033[94m-- Loading data\033[0m")
    data_loader = ImageDataLoader(
        input_dir=input_dir,
        target_dir=target_dir,
        batch_size=batch_size,
        train_pct=train_pct,
        valid_pct=valid_pct,
        device=device
    )

    print("\033[94m-- Calculating or loading class weights\033[0m")
    train_stats = data_loader.count_classes(n_classes)
    class_weights = get_class_weights(train_stats)

    print("\033[94m-- Initialising model, loss function, and optimiser\033[0m")
    model, loss_fn, optim = create_model(n_classes, class_weights, device)
    model = model.to(device)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    output_path = os.path.join(output_dir, f"{model_name}_metrics.root")
    with uproot.recreate(output_path) as output:
        print("\033[94m-- Starting training loop\033[0m")
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(n_epochs):
                model.train()
                print(f"\033[96m-- Epoch {epoch+1}/{n_epochs} started\033[0m")

                batch_train_loss = []

                for i, batch in enumerate(data_loader.train_dl):
                    x, y = batch
                    print(x.sum())
                    x, y = x.to(device), y.to(device)

                    if torch.isnan(x).any() or torch.isinf(x).any():
                        print(f"-- NaN or Inf detected in input data at epoch {epoch}, batch {i}")
                        print(f"x min: {x.min().item()}, x max: {x.max().item()}, x mean: {x.mean().item()}")
                        return

                    optim.zero_grad()

                    y = y.to(torch.long)

                    with torch.autograd.set_detect_anomaly(True):
                        with torch.amp.autocast('cuda'):
                            pred = model(x)
                            print(pred)

                            if torch.isnan(pred).any() or torch.isinf(pred).any():
                                print(f"-- NaN or Inf detected in model predictions at epoch {epoch}, batch {i}")
                                print(f"pred min: {pred.min().item()}, pred max: {pred.max().item()}, pred mean: {pred.mean().item()}")
                                return

                            loss = loss_fn(pred, y)
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                print(f"-- NaN or Inf detected in loss at epoch {epoch}, batch {i}")
                                print(f"loss value: {loss.item()}")
                                return

                        scaler.scale(loss).backward()

                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"-- NaN or Inf detected in gradients of {name} at epoch {epoch}, batch {i}")
                                return

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optim)
                    scaler.update()

                    batch_train_loss.append(loss.item())
                    print(f"\033[96m--- Epoch {epoch+1}, Batch {i+1} - Training Loss: {loss.item()}\033[0m")

                train_loss_mean = np.mean(batch_train_loss)
                train_loss_std_dev = np.std(batch_train_loss)
                print(f"Epoch {epoch+1} - Mean Training Loss: {train_loss_mean}, Std Dev: {train_loss_std_dev}")

                model.eval()
                batch_valid_loss = []
                with torch.no_grad():
                    for j, batch in enumerate(data_loader.valid_dl):
                        x, y = batch
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        val_loss = loss_fn(pred, y)
                        batch_valid_loss.append(val_loss.item())

                valid_loss_mean = np.mean(batch_valid_loss)
                valid_loss_std_dev = np.std(batch_valid_loss)
                print(f"Epoch {epoch+1} - Mean Validation Loss: {valid_loss_mean}, Std Dev: {valid_loss_std_dev}")

                scheduler.step(valid_loss_mean)

                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} - Current Learning Rate: {current_lr}")

            output["metrics"] = {
                "train_loss": train_loss,
                "train_loss_std": train_loss_std,
                "valid_loss": valid_loss,
                "valid_loss_std": valid_loss_std,
                "learning_rate": learning_rates,
            }   

    print("\033[94m-- Training complete\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    
    config = ConfigLoader(args.config) 
    train(config)