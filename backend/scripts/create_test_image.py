#!/usr/bin/env python3
"""
Create Test Image for Breast Cancer Classification Testing
Generates a simple test image for prediction testing
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_test_image(filename="test_image.png", size=(224, 224)):
    """
    Create a simple test image for prediction testing
    """
    # Create a simple grayscale image that looks like a mammogram
    # This is just for testing purposes - not a real mammogram

    # Create base image with some structure
    img = np.random.normal(0.5, 0.1, size + (3,))

    # Add some circular structures to simulate breast tissue
    center_y, center_x = size[0] // 2, size[1] // 2

    # Create circular patterns
    y, x = np.ogrid[: size[0], : size[1]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(size) // 3) ** 2

    # Add some texture
    img[mask] += 0.2 * np.random.normal(0, 1, mask.sum())

    # Ensure values are in [0, 1] range
    img = np.clip(img, 0, 1)

    # Convert to PIL Image
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    # Save the image
    img_pil.save(filename)
    print(f"✅ Test image created: {filename}")
    print(f"  Size: {size}")
    print(f"  Format: PNG")

    return filename


def create_multiple_test_images():
    """
    Create multiple test images with different characteristics
    """
    test_images = []

    # Test image 1: Simple structure
    filename1 = "test_image_1.png"
    create_test_image(filename1, (224, 224))
    test_images.append(filename1)

    # Test image 2: Different pattern
    filename2 = "test_image_2.png"
    img = np.random.normal(0.6, 0.15, (224, 224, 3))
    img_pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img_pil.save(filename2)
    print(f"✅ Test image created: {filename2}")
    test_images.append(filename2)

    # Test image 3: High contrast
    filename3 = "test_image_3.png"
    img = np.random.normal(0.3, 0.2, (224, 224, 3))
    img_pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    img_pil.save(filename3)
    print(f"✅ Test image created: {filename3}")
    test_images.append(filename3)

    return test_images


def main():
    """
    Main function
    """
    print("=" * 50)
    print("🖼️ Creating Test Images for Breast Cancer Classification")
    print("=" * 50)

    # Create single test image
    print("\n📁 Creating single test image...")
    test_image = create_test_image()

    # Create multiple test images
    print("\n📁 Creating multiple test images...")
    test_images = create_multiple_test_images()

    print(f"\n✅ Created {len(test_images) + 1} test images:")
    print(f"  - {test_image}")
    for img in test_images:
        print(f"  - {img}")

    print(f"\n💡 You can now test the prediction system with:")
    print(f"  python predict_single_image.py {test_image}")
    print(f"  python test_prediction_system.py")


if __name__ == "__main__":
    main()
