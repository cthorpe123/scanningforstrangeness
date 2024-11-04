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
    loss_fn = FocalLoss(alpha=torch.as_tensor(weights, device=device, dtype=torch.float), gamma=2.0)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, loss_fn, optim

def get_class_weights(stats):
    stats[stats == 0.] = 1.
    weights = 1. / stats
    return weights / weights.sum()

def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        y_prediction = model(x)
        loss = loss_fn(y_prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_step

def make_test_step(model, loss_fn):
    def test_step(loader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_prediction = model(x_batch)
                loss = loss_fn(y_prediction, y_batch)
                total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        return avg_loss
    return test_step

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)

    batch_size = config.batch_size
    train_pct = config.train_pct
    valid_pct = config.valid_pct

    n_classes = config.n_classes
    n_epochs = config.n_epochs

    input_dir = config.input_dir
    target_dir = config.target_dir
    output_dir = config.output_dir

    model_name = config.model_name

    data_loader = ImageDataLoader(
        input_dir=input_dir,
        target_dir=target_dir,
        batch_size=batch_size,
        train_pct=train_pct,
        valid_pct=valid_pct,
        device=device
    )

    train_stats = data_loader.count_classes(n_classes)
    class_weights = get_class_weights(torch.tensor(train_stats, dtype=torch.float32).to(device))

    model, loss_fn, optim = create_model(n_classes, class_weights, device)
    model = model.to(device)

    metrics_tree = {
        "epoch": [],
        "step": [],
        "train_loss": [],
        "valid_loss": []
    }

    train_step = make_train_step(model, loss_fn, optim)
    test_step = make_test_step(model, loss_fn)

    output_path = os.path.join(output_dir, f"{model_name}_metrics.root")
    output = uproot.recreate(output_path)

    step = 0
    for epoch in range(n_epochs):
        model.train()
        for x, y in data_loader.train_dl:
            x, y = x.to(device), y.to(device)
            train_loss = train_step(x, y)
            valid_loss = test_step(data_loader.valid_dl, device)

            metrics_tree["epoch"].append(epoch)
            metrics_tree["step"].append(step)
            metrics_tree["train_loss"].append(train_loss)
            metrics_tree["valid_loss"].append(valid_loss)

            model_save_path = os.path.join(output_dir, f"{model_name}_epoch{epoch}_step{step}.pt")
            torch.save(model.state_dict(), model_save_path)

            print(f"\033[93m-- Epoch [{epoch+1}/{n_epochs}], Step {step}: "
                  f"-- Train Loss: {train_loss:.4f}, "
                  f"-- Valid Loss: {valid_loss:.4f}\033[0m")
            print(f"\033[92m-- Model saved to {model_save_path}\033[0m")
            
            step += 1

    output["metrics"] = metrics_tree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = ConfigLoader(args.config) 
    train(config)