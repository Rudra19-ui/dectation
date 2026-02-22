#!/usr/bin/env python3
"""
Dataset Analysis Script
This script analyzes the breast cancer dataset and counts samples for each class
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_dataset():
    """Analyze the dataset and count classes"""
    print("🔍 Analyzing Breast Cancer Dataset")
    print("=" * 50)

    # Check if data directory exists
    data_dir = "data"
    images_dir = "data/images"

    if not os.path.exists(data_dir):
        print("❌ Data directory not found!")
        print("Expected structure:")
        print("  data/")
        print("  ├── images/")
        print("  ├── train.csv")
        print("  ├── val.csv")
        print("  └── test.csv")
        return

    if not os.path.exists(images_dir):
        print("❌ Images directory not found!")
        print("Expected: data/images/")
        return

    # Check for CSV files
    csv_files = {
        "train": "data/train.csv",
        "val": "data/val.csv",
        "test": "data/test.csv",
    }

    missing_files = []
    for split, file_path in csv_files.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing CSV files:")
        for file in missing_files:
            print(f"  - {file}")
        return

    # Analyze each split
    total_stats = {}

    for split_name, csv_path in csv_files.items():
        print(f"\n📊 Analyzing {split_name} split...")

        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {csv_path}")
            print(f"   Shape: {df.shape}")

            # Check required columns
            required_cols = ["filename", "label"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                print(f"   Available columns: {list(df.columns)}")
                continue

            # Count labels
            label_counts = df["label"].value_counts()
            print(f"   Label distribution:")
            for label, count in label_counts.items():
                print(f"     {label}: {count}")

            # Check if images exist
            missing_images = []
            for filename in df["filename"]:
                image_path = os.path.join(images_dir, filename)
                if not os.path.exists(image_path):
                    missing_images.append(filename)

            if missing_images:
                print(f"⚠️  Missing images: {len(missing_images)}")
                print(f"   Total images in CSV: {len(df)}")
                print(f"   Available images: {len(df) - len(missing_images)}")
            else:
                print(f"✅ All images found!")

            total_stats[split_name] = {
                "total": len(df),
                "label_counts": label_counts.to_dict(),
                "missing_images": len(missing_images),
            }

        except Exception as e:
            print(f"❌ Error reading {csv_path}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📋 DATASET SUMMARY")
    print("=" * 50)

    if total_stats:
        # Overall counts
        all_labels = set()
        for split_stats in total_stats.values():
            all_labels.update(split_stats["label_counts"].keys())

        print(f"Classes found: {sorted(all_labels)}")
        print()

        # Per split summary
        for split_name, stats in total_stats.items():
            print(f"{split_name.upper()} SPLIT:")
            print(f"  Total samples: {stats['total']}")
            print(f"  Missing images: {stats['missing_images']}")
            print(f"  Available images: {stats['total'] - stats['missing_images']}")
            print(f"  Label distribution:")
            for label in sorted(all_labels):
                count = stats["label_counts"].get(label, 0)
                print(f"    {label}: {count}")
            print()

        # Total counts across all splits
        print("TOTAL ACROSS ALL SPLITS:")
        total_by_class = Counter()
        total_samples = 0
        total_missing = 0

        for split_stats in total_stats.values():
            total_samples += split_stats["total"]
            total_missing += split_stats["missing_images"]
            for label, count in split_stats["label_counts"].items():
                total_by_class[label] += count

        for label in sorted(all_labels):
            print(f"  {label}: {total_by_class[label]}")

        print(f"  Total samples: {total_samples}")
        print(f"  Missing images: {total_missing}")
        print(f"  Available images: {total_samples - total_missing}")

        # Create visualization
        try:
            create_visualization(total_stats, all_labels)
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

    else:
        print("❌ No valid data found!")


def create_visualization(stats, labels):
    """Create a visualization of the dataset distribution"""
    print("\n📈 Creating visualization...")

    # Prepare data for plotting
    splits = list(stats.keys())
    label_names = sorted(labels)

    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar chart of total samples per split
    totals = [stats[split]["total"] for split in splits]
    ax1.bar(splits, totals, color=["#ff9999", "#66b3ff", "#99ff99"])
    ax1.set_title("Total Samples per Split")
    ax1.set_ylabel("Number of Samples")
    ax1.set_ylim(0, max(totals) * 1.1)

    # Add value labels on bars
    for i, v in enumerate(totals):
        ax1.text(i, v + max(totals) * 0.01, str(v), ha="center", va="bottom")

    # Stacked bar chart of class distribution
    x = np.arange(len(splits))
    width = 0.8

    bottom = np.zeros(len(splits))
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

    for i, label in enumerate(label_names):
        values = [stats[split]["label_counts"].get(label, 0) for split in splits]
        ax2.bar(
            splits, values, bottom=bottom, label=label, color=colors[i % len(colors)]
        )
        bottom += values

    ax2.set_title("Class Distribution per Split")
    ax2.set_ylabel("Number of Samples")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Visualization saved as 'dataset_analysis.png'")


def create_sample_data():
    """Create sample data structure for testing"""
    print("\n🔧 Creating sample data structure...")

    # Create directories
    os.makedirs("data/images", exist_ok=True)

    # Sample data structure
    sample_data = {
        "train": [
            {"filename": "normal_001.png", "label": "normal"},
            {"filename": "normal_002.png", "label": "normal"},
            {"filename": "benign_001.png", "label": "benign"},
            {"filename": "benign_002.png", "label": "benign"},
            {"filename": "malignant_001.png", "label": "malignant"},
        ],
        "val": [
            {"filename": "normal_003.png", "label": "normal"},
            {"filename": "benign_003.png", "label": "benign"},
            {"filename": "malignant_002.png", "label": "malignant"},
        ],
        "test": [
            {"filename": "normal_004.png", "label": "normal"},
            {"filename": "benign_004.png", "label": "benign"},
            {"filename": "malignant_003.png", "label": "malignant"},
        ],
    }

    # Create CSV files
    for split, data in sample_data.items():
        df = pd.DataFrame(data)
        csv_path = f"data/{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Created {csv_path}")

    print("\n📋 Sample data structure created:")
    print("  - data/images/ (empty directory for images)")
    print("  - data/train.csv (5 samples)")
    print("  - data/val.csv (3 samples)")
    print("  - data/test.csv (3 samples)")
    print("\n⚠️  Note: You need to add actual mammogram images to data/images/")


if __name__ == "__main__":
    analyze_dataset()

    # If no data found, offer to create sample structure
    if not os.path.exists("data"):
        print("\n" + "=" * 50)
        print("Would you like to create a sample data structure?")
        print("This will create the expected directories and CSV files.")
        print("You can then add your mammogram images to data/images/")

        # For now, let's create the sample structure
        create_sample_data()
