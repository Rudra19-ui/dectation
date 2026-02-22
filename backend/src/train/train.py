import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.baseline_cnn import BaselineCNN
from models.transfer_learning import get_transfer_model
from src.data.dataset import get_dataloaders
from utils.metrics import compute_metrics

# Config
DATA_DIR = "data/images"  # Update as needed
SPLIT_CSVS = {"train": "data/train.csv", "val": "data/val.csv", "test": "data/test.csv"}
MODEL_TYPE = "resnet50"  # 'baseline', 'resnet50', 'vgg16', 'efficientnet_b0'
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/breast_cancer_detector.pt"

# Data
dataloaders = get_dataloaders(DATA_DIR, SPLIT_CSVS, batch_size=BATCH_SIZE)

# Model
if MODEL_TYPE == "baseline":
    model = BaselineCNN(num_classes=NUM_CLASSES)
    target_layer = model.conv2
else:
    model, target_layer = get_transfer_model(MODEL_TYPE, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# Training loop
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.float().to(DEVICE)
            labels = labels.long().to(DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        metrics = compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES)
        print(
            f'{phase} Loss: {epoch_loss:.4f} | Acc: {metrics["accuracy"]:.4f} | F1: {metrics["f1"]:.4f}'
        )
        if phase == "val":
            scheduler.step(epoch_loss)
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print("Model saved!")
print("Training complete.")
