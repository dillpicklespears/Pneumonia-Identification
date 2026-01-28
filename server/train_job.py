import os
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from model import PneumoniaCNN

COMBINED_TRAIN_DIR = os.path.join("server", "storage", "combined_train")
MODELS_DIR = os.path.join("server", "models", "versions")
ACTIVE_MODEL = os.path.join("server", "models", "active.pth")

def train_and_publish(num_epochs: int = 2, batch_size: int = 32, lr: float = 1e-4):
    if not os.path.isdir(COMBINED_TRAIN_DIR):
        raise FileNotFoundError(f"Missing combined train dir: {COMBINED_TRAIN_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(COMBINED_TRAIN_DIR, transform=tfm)

    # Simple train/val split
    val_ratio = 0.2
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PneumoniaCNN().to(device)

    # Class weights (optional): ImageFolder gives class_to_idx; match your mapping
    # NORMAL=0, PNEUMONIA=1 (should match folder names)
    weights = torch.tensor([1.94, 0.67], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total if v_total else 0.0

        print(f"Epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # Save versioned model (full model object like your current setup)
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    version_path = os.path.join(MODELS_DIR, f"model_{ts}.pth")

    torch.save(model, version_path)

    # Publish (swap active)
    shutil.copy2(version_path, ACTIVE_MODEL)

    return {
        "version_path": version_path,
        "active_path": ACTIVE_MODEL,
        "best_val_acc": best_val_acc,
        "device": str(device),
        "train_samples": train_len,
        "val_samples": val_len,
        "classes": dataset.classes,
    }
