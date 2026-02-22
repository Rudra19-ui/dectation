#!/usr/bin/env python3
"""
Comprehensive Dataset Preparation Script
This script prepares the CBIS-DDSM dataset for training using available metadata
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split


def create_synthetic_mammogram(label, size=(224, 224)):
    """Create synthetic mammogram images for training"""
    # Create base image (dark background for mammogram-like appearance)
    img = Image.new("L", size, color=20)  # Dark gray background

    # Add some random noise to simulate mammogram texture
    noise = np.random.normal(30, 10, size)
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise)

    # Blend with base image
    img = Image.blend(img, noise_img, 0.7)

    # Add features based on label
    draw = ImageDraw.Draw(img)

    if label == "malignant":
        # Add irregular, spiculated mass-like features
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(15, 25)

        # Draw irregular mass
        points = []
        for i in range(8):
            angle = i * 45 + random.randint(-15, 15)
            r = radius + random.randint(-5, 10)
            x = center_x + int(r * np.cos(np.radians(angle)))
            y = center_y + int(r * np.sin(np.radians(angle)))
            points.append((x, y))

        # Fill with darker color (mass)
        draw.polygon(points, fill=60)

        # Add spiculations
        for _ in range(6):
            angle = random.uniform(0, 360)
            length = random.randint(8, 15)
            start_x = center_x + int(radius * np.cos(np.radians(angle)))
            start_y = center_y + int(radius * np.sin(np.radians(angle)))
            end_x = start_x + int(length * np.cos(np.radians(angle)))
            end_y = start_y + int(length * np.sin(np.radians(angle)))
            draw.line([(start_x, start_y), (end_x, end_y)], fill=40, width=2)

    elif label == "benign":
        # Add smooth, well-defined mass-like features
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(20, 30)

        # Draw smooth circular mass
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ]
        draw.ellipse(bbox, fill=80, outline=100)

    else:  # normal
        # Add normal breast tissue patterns
        for _ in range(10):
            x = random.randint(0, size[0])
            y = random.randint(0, size[1])
            radius = random.randint(2, 8)
            intensity = random.randint(40, 80)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius], fill=intensity
            )

    # Convert to RGB
    img_rgb = img.convert("RGB")
    return img_rgb


def process_csv_to_dataset(csv_file, output_dir, num_samples_per_class=100):
    """Process CSV file and create synthetic dataset"""
    print(f"Processing {csv_file}...")

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Extract pathology information
    if "pathology" in df.columns:
        # Map pathology to our classes
        pathology_mapping = {
            "MALIGNANT": "malignant",
            "BENIGN": "benign",
            "BENIGN_WITHOUT_CALLBACK": "benign",
        }

        df["label"] = df["pathology"].map(pathology_mapping)
        df = df.dropna(subset=["label"])

        # Count samples per class
        label_counts = df["label"].value_counts()
        print(f"Original label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

        # Sample equal numbers from each class
        sampled_data = []
        for label in ["malignant", "benign"]:
            if label in df["label"].values:
                label_df = df[df["label"] == label]
                if len(label_df) > num_samples_per_class:
                    sampled = label_df.sample(n=num_samples_per_class, random_state=42)
                else:
                    sampled = label_df
                sampled_data.append(sampled)

        if sampled_data:
            df = pd.concat(sampled_data, ignore_index=True)
        else:
            # If no pathology data, create synthetic data
            df = pd.DataFrame(
                {
                    "label": ["malignant"] * num_samples_per_class
                    + ["benign"] * num_samples_per_class
                    + ["normal"] * num_samples_per_class
                }
            )

    else:
        # Create synthetic data if no pathology column
        df = pd.DataFrame(
            {
                "label": ["malignant"] * num_samples_per_class
                + ["benign"] * num_samples_per_class
                + ["normal"] * num_samples_per_class
            }
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic images
    dataset_data = []
    for idx, row in df.iterrows():
        label = row["label"]

        # Create synthetic image
        img = create_synthetic_mammogram(label)

        # Save image
        filename = f"{label}_{idx:04d}.png"
        img_path = os.path.join(output_dir, filename)
        img.save(img_path)

        dataset_data.append({"filename": filename, "label": label})

    return pd.DataFrame(dataset_data)


def create_dataset_splits(df, output_dir):
    """Create train/val/test splits"""
    print("Creating dataset splits...")

    # Use stratified split
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )

    # Save CSV files
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Dataset splits created:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Show label distribution
    print("\nLabel distribution:")
    for split_name, split_df in [
        ("Train", train_df),
        ("Val", val_df),
        ("Test", test_df),
    ]:
        print(f"  {split_name}:")
        label_counts = split_df["label"].value_counts()
        for label, count in label_counts.items():
            print(f"    {label}: {count}")

    return train_df, val_df, test_df


def create_dataset_visualization(train_df, val_df, test_df):
    """Create dataset visualization"""
    print("Creating dataset visualization...")

    # Combine all splits
    all_data = []
    for df, split in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        for _, row in df.iterrows():
            all_data.append({"split": split, "label": row["label"]})

    viz_df = pd.DataFrame(all_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Label distribution
    label_counts = viz_df["label"].value_counts()
    ax1.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%")
    ax1.set_title("Overall Label Distribution")

    # Split distribution
    split_label_counts = viz_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    split_label_counts.plot(kind="bar", ax=ax2)
    ax2.set_title("Label Distribution by Split")
    ax2.set_xlabel("Dataset Split")
    ax2.set_ylabel("Number of Samples")
    ax2.legend(title="Label")
    ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches="tight")
    print("✅ Dataset visualization saved as dataset_analysis.png")


def main():
    """Main dataset preparation function"""
    print("🏥 CBIS-DDSM Dataset Preparation")
    print("=" * 50)

    # Configuration
    output_dir = "data/images"
    num_samples_per_class = 200  # Adjust based on your needs

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Check for existing CSV files
    csv_files = [
        "dataset4/calc_case_description_train_set.csv",
        "dataset4/mass_case_description_train_set.csv",
    ]

    available_csv = None
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            available_csv = csv_file
            break

    if available_csv:
        print(f"✅ Found CSV file: {available_csv}")
        # Process the CSV file
        df = process_csv_to_dataset(available_csv, output_dir, num_samples_per_class)
    else:
        print("⚠️ No CSV files found, creating synthetic dataset...")
        # Create synthetic dataset
        df = process_csv_to_dataset(None, output_dir, num_samples_per_class)

    print(f"✅ Created {len(df)} synthetic images")

    # Create dataset splits
    train_df, val_df, test_df = create_dataset_splits(df, output_dir)

    # Create visualization
    create_dataset_visualization(train_df, val_df, test_df)

    print("\n🎉 Dataset preparation complete!")
    print("📋 Next steps:")
    print("1. Run training: python train_model.py")
    print("2. Or run sample training: python train_sample.py")
    print("3. Test the web app: streamlit run webapp/streamlit_app.py")


if __name__ == "__main__":
    main()
