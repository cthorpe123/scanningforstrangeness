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
from tqdm import tqdm 

from lib.dataset import ImageDataLoader
from lib.model import UNet
from lib.loss import FocalLoss
from lib.common import set_seed
from lib.config import ConfigLoader

from sklearn.metrics import precision_score, recall_score

def create_model(n_classes, weights, device):
    model = UNet(1, n_classes=n_classes, depth=4, n_filters=16)
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

def save_class_weights(weights, path="class_weights.npy"):
    np.save(path, weights)
    print("\033[34m-- Class weights saved\033[0m")

def load_class_weights(path="class_weights.npy"):
    if os.path.exists(path):
        print("\033[34m-- Loading saved class weights\033[0m")
        return np.load(path)
    else:
        print("\033[35m-- Class weights file not found; calculating class weights\033[0m")
        return None

def calculate_metrics(pred, target, n_classes):
    pred = pred.argmax(dim=1).flatten().cpu().numpy()  
    target = target.flatten().cpu().numpy()  

    accuracy = (pred == target).mean()
    precision = precision_score(target, pred, labels=[1, 2], average='macro', zero_division=0)
    recall = recall_score(target, pred, labels=[1, 2], average='macro', zero_division=0)

    return accuracy, precision, recall

def train(config):
    print("\033[34m-- Initialising training configuration\033[0m")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    print(f"\033[34m-- Using device: {device}\033[0m")

    batch_size = config.batch_size
    train_pct = config.train_pct
    valid_pct = config.valid_pct
    n_classes = config.n_classes
    n_epochs = config.n_epochs

    input_dir = config.input_dir
    target_dir = config.target_dir
    output_dir = config.output_dir
    model_name = config.model_name
    model_save_dir = os.path.join(output_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    print("\033[34m-- Loading data\033[0m")
    data_loader = ImageDataLoader(
        input_dir=input_dir,
        target_dir=target_dir,
        batch_size=batch_size,
        train_pct=train_pct,
        valid_pct=valid_pct,
        device=device
    )

    print("\033[34m-- Loading or calculating class weights\033[0m")
    class_weights = load_class_weights()  
    if class_weights is None:  
        train_stats = data_loader.count_classes(n_classes)
        class_weights = get_class_weights(train_stats)
        save_class_weights(class_weights)

    print("\033[34m-- Initialising model, loss function, and optimiser\033[0m")
    model, loss_fn, optim = create_model(n_classes, class_weights, device)
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    train_losses, train_loss_stds = [], []
    valid_signature_losses, valid_signature_loss_stds = [], []
    valid_background_losses, valid_background_loss_stds = [], []
    learning_rates = []
    signature_metrics, background_metrics = [], []

    output_path = os.path.join(output_dir, f"{model_name}_metrics.root")
    with uproot.recreate(output_path) as output:
        print("\033[34m-- Starting training loop\033[0m")
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(n_epochs):
                model.train()
                print(f"\033[34m-- Epoch {epoch+1}/{n_epochs} started\033[0m")

                batch_train_loss = []
                for i, batch in enumerate(tqdm(data_loader.train_dl, desc=f"Training Epoch {epoch+1}")):
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                    y = y.to(torch.long)
                    optim.zero_grad()

                    pred = model(x)
                    loss = loss_fn(pred, y)
                        
                    loss.backward()
                    optim.step()

                    batch_train_loss.append(loss.item())
                    tqdm.write(f"\033[34m--- Epoch {epoch+1}, Batch {i+1} - Training Loss: {loss.item()}\033[0m")

                train_loss_mean = np.mean(batch_train_loss)
                train_loss_std_dev = np.std(batch_train_loss)
                print(f"Epoch {epoch+1} - Mean Training Loss: {train_loss_mean}, Std Dev: {train_loss_std_dev}")

                train_losses.append(train_loss_mean)
                train_loss_stds.append(train_loss_std_dev)

                model.eval()
                signature_loss = []
                signature_acc, signature_precision, signature_recall = [], [], []
                with torch.no_grad():
                    for batch in tqdm(data_loader.valid_signature_dl, desc=f"Signature Validation Epoch {epoch+1}"):
                        x, y = batch
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        val_loss = loss_fn(pred, y)
                        signature_loss.append(val_loss.item())

                        acc, prec, rec = calculate_metrics(pred, y, n_classes)
                        signature_acc.append(acc)
                        signature_precision.append(prec)
                        signature_recall.append(rec)

                valid_signature_loss_mean = np.mean(signature_loss)
                valid_signature_loss_std = np.std(signature_loss)
                valid_signature_losses.append(valid_signature_loss_mean)
                valid_signature_loss_stds.append(valid_signature_loss_std)
                print(f"Epoch {epoch+1} - Mean Signature Validation Loss: {valid_signature_loss_mean}, Std Dev: {valid_signature_loss_std}")

                signature_metrics.append({
                    "accuracy_mean": np.mean(signature_acc),
                    "accuracy_std": np.std(signature_acc),
                    "precision_mean": np.mean(signature_precision),
                    "precision_std": np.std(signature_precision),
                    "recall_mean": np.mean(signature_recall),
                    "recall_std": np.std(signature_recall),
                })

                background_loss = []
                background_acc, background_precision, background_recall = [], [], []
                with torch.no_grad():
                    for batch in tqdm(data_loader.valid_background_dl, desc=f"Background Validation Epoch {epoch+1}"):
                        x, y = batch
                        x, y = x.to(device), y.to(device)

                        pred = model(x)
                        val_loss = loss_fn(pred, y)
                        background_loss.append(val_loss.item())

                        acc, prec, rec = calculate_metrics(pred, y, n_classes)
                        background_acc.append(acc)
                        background_precision.append(prec)
                        background_recall.append(rec)

                background_metrics.append({
                    "accuracy_mean": np.mean(background_acc),
                    "accuracy_std": np.std(background_acc),
                    "precision_mean": np.mean(background_precision),
                    "precision_std": np.std(background_precision),
                    "recall_mean": np.mean(background_recall),
                    "recall_std": np.std(background_recall),
                })

                valid_background_loss_mean = np.mean(background_loss)
                valid_background_loss_std = np.std(background_loss)
                valid_background_losses.append(valid_background_loss_mean)
                valid_background_loss_stds.append(valid_background_loss_std)
                print(f"Epoch {epoch+1} - Mean Background Validation Loss: {valid_background_loss_mean}, Std Dev: {valid_background_loss_std}")

                avg_validation_loss = (valid_signature_loss_mean + valid_background_loss_mean) / 2
                scheduler.step(avg_validation_loss)
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)
                print(f"Epoch {epoch+1} - Current Learning Rate: {current_lr}")

                model_save_path = os.path.join(model_save_dir, f"{model_name}_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), model_save_path)
                print(f"\033[32m-- Model saved at {model_save_path}\033[0m")

            output["metrics"] = {
                "train_loss": train_losses,
                "train_loss_std": train_loss_stds,
                "valid_signature_loss": valid_signature_losses,
                "valid_signature_loss_std": valid_signature_loss_stds,
                "valid_background_loss": valid_background_losses,
                "valid_background_loss_std": valid_background_loss_stds,
                "learning_rate": learning_rates,
                "signature_metrics": signature_metrics,
                "background_metrics": background_metrics,
            }   

    print("\033[32m-- Training complete\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    
    config = ConfigLoader(args.config) 
    train(config)
