import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import load_dataloader, load_model, set_random_seed, CLEAN_MODEL_DIR
from utils import FP_MAP


def get_optimizer_and_scheduler(
    model: nn.Module, task_id: int, num_epochs: int
):
    """
    Create optimizer and scheduler based on task_id.

    Args:
        model: The model to optimize.
        task_id: The task identifier that determines optimizer choice.
        num_epochs: Default number of epochs.

    Returns:
        optimizer, scheduler, num_epochs (may be adjusted for specific tasks).
    """
    if task_id in (6, 7):
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 1  # override for specific tasks
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        num_epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler, num_epochs


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Returns:
        Training accuracy (0–1).
    """
    model.train()
    correct, total = 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, y = batch["input"].to(device), batch["label"].to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)

    return correct / total


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """
    Evaluate the model on a dataset.

    Returns:
        Accuracy (0–1).
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["input"].to(device), batch["label"].to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def save_best_model(model: nn.Module, fp, acc: float, best_acc: float) -> float:
    if acc > best_acc:
        if fp == "fp32":
            save_path = os.path.join(CLEAN_MODEL_DIR, f"{model.model_data_name}_best.pth")
        else:
            save_path = os.path.join(CLEAN_MODEL_DIR, f"{model.model_data_name}_{fp}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"New best accuracy: {acc:.4f} -> Saved to {save_path}")
        return acc
    return best_acc


def main(task_id: int, fp_name: str) -> None:
    """
    Main training routine.
    """
    # Hyperparameters
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and model
    train_loader, valid_loader, test_loader = load_dataloader(
        task_id, is_shuffle=True, train_batch=100, test_batch=200
    )
    fp = FP_MAP[fp_name]
    model = load_model(task_id, load_pretrained=False)
    model = model.to(fp)

    # Add dropout if missing
    if not hasattr(model, "dropout"):
        model.add_module("dropout", nn.Dropout(p=0.25))

    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler, num_epochs = get_optimizer_and_scheduler(model, task_id, num_epochs)

    print(f"Training {model.model_data_name} on task {task_id}")
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Acc: {train_acc:.4f}")

        # Periodic evaluation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Test Accuracy: {test_acc * 100:.2f}%")
            best_acc = save_best_model(model, fp_name, test_acc, best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="Task ID for training")
    parser.add_argument("--fp", type=str, default="fp32", help="FP mode")
    args = parser.parse_args()

    assert args.fp in ["fp32", "fp16", "fp64"]

    set_random_seed(3407)
    main(task_id=args.task_id, fp_name=args.fp)
