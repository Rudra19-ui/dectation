#!/usr/bin/env python3
"""
Simplified Training Script for Sample Dataset
This script trains the model with the sample CBIS-DDSM data
"""

import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.baseline_cnn import BaselineCNN
from models.transfer_learning import get_transfer_model

# Import project modules
from src.data.dataset import get_dataloaders
from utils.metrics import compute_metrics


def train_sample_model():
    """Train model with sample data"""
    print("🏥 Sample Breast Cancer Detection Training")
    print("=" * 50)

    # Configuration for small dataset
    MODEL_TYPE = "baseline"  # Use baseline for small dataset
    BATCH_SIZE = 2  # Small batch size for small dataset
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Check data
    if not os.path.exists("data/train.csv"):
        print("❌ Training data not found!")
        print("Please run: python process_sample_cbisddsm.py")
        return None, None

    print(f"📋 Training Configuration:")
    print(f"   Model Type: {MODEL_TYPE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")

    # Load data
    try:
        print("\n📦 Loading data...")
        dataloaders = get_dataloaders(
            "data/images",
            {"train": "data/train.csv", "val": "data/val.csv", "test": "data/test.csv"},
            batch_size=BATCH_SIZE,
        )
        print("✅ Data loaded successfully")

        for phase, dataloader in dataloaders.items():
            print(
                f"   {phase}: {len(dataloader)} batches, {len(dataloader.dataset)} samples"
            )

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

    # Create model
    print(f"\n🏗️ Creating {MODEL_TYPE} model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_TYPE == "baseline":
        model = BaselineCNN(num_classes=3)
    else:
        model, _ = get_transfer_model(MODEL_TYPE, num_classes=3)

    model = model.to(device)
    print(f"✅ Model created and moved to {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🎯 Starting Training for {EPOCHS} epochs...")
    print("=" * 50)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            with torch.set_grad_enabled(phase == "train"):
                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = 100 * correct / total

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc)
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc)

            print(
                f"   {phase.capitalize()}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%"
            )

            # Save best model
            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(
                    model.state_dict(), "models/breast_cancer_detector_sample.pt"
                )
                print(f"   💾 Saved best model (val_loss: {epoch_loss:.4f})")

    print("\n🎉 Training Complete!")
    print("📊 Final Results:")
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    print(f"   Final Training Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")

    # Create simple training plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Validation Loss", color="red")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy", color="blue")
    plt.plot(history["val_acc"], label="Validation Accuracy", color="red")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sample_training_history.png", dpi=300, bbox_inches="tight")
    print("✅ Training plot saved as: sample_training_history.png")

    return model, history


def main():
    """Main function"""
    model, history = train_sample_model()

    if model is not None:
        print("\n📋 Next Steps:")
        print("1. Test the web app: python -m streamlit run webapp/streamlit_app.py")
        print("2. Test the API: python api/app.py")
        print(
            "3. The trained model is saved as: models/breast_cancer_detector_sample.pt"
        )
    else:
        print("\n❌ Training failed. Please check the data and try again.")


if __name__ == "__main__":
    main()
