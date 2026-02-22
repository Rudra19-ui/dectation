#!/usr/bin/env python3
"""
Windows Configuration Checker for TensorFlow Training
Validates that all settings are Windows-safe before training
"""

import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def check_tensorflow_configuration():
    """
    Check TensorFlow configuration for Windows compatibility
    """
    print("🔧 Checking TensorFlow Configuration...")

    # Check TensorFlow version
    tf_version = tf.__version__
    print(f"📦 TensorFlow version: {tf_version}")

    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        print(f"🎮 GPU detected: {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("💻 No GPU detected, using CPU")

    # Check environment variables
    env_vars = {
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "Not set"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "Not set"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "Not set"),
    }

    print("\n🔍 Environment Variables:")
    for var, value in env_vars.items():
        print(f"  {var}: {value}")

    return True


def check_dataset_structure(dataset_path):
    """
    Check dataset structure and file counts
    """
    print(f"\n📁 Checking dataset structure: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return False

    expected_folders = ["benign", "malignant", "normal"]
    total_files = 0

    for folder in expected_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Missing folder: {folder}")
            return False

        # Count image files
        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"✅ {folder}: {len(image_files)} images")
        total_files += len(image_files)

    print(f"📊 Total images: {total_files}")

    if total_files < 10:
        print("⚠️ Warning: Very few images detected. Training may not be effective.")

    return True


def test_data_generator_configuration(
    dataset_path, batch_size=8, input_size=(224, 224)
):
    """
    Test data generator with Windows-safe settings
    """
    print(f"\n🧪 Testing Data Generator Configuration...")
    print(f"📦 Batch size: {batch_size}")
    print(f"🖼️ Input size: {input_size}")

    try:
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Test with Windows-safe settings
        test_generator = test_datagen.flow_from_directory(
            dataset_path,
            target_size=input_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            workers=0,  # CRITICAL: No multiprocessing
            use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
            max_queue_size=10,  # Reduced queue size
        )

        print(f"✅ Data generator created successfully!")
        print(f"📊 Samples: {test_generator.samples}")
        print(f"📊 Classes: {test_generator.class_indices}")

        # Test getting one batch
        print("🧪 Testing batch retrieval...")
        batch = next(test_generator)
        print(f"✅ Batch retrieved successfully!")
        print(f"📊 Batch shape: {batch[0].shape}")
        print(f"📊 Labels shape: {batch[1].shape}")

        return True

    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        return False


def check_memory_requirements(batch_size, input_size, num_classes=3):
    """
    Estimate memory requirements
    """
    print(f"\n💾 Checking Memory Requirements...")

    # Calculate approximate memory usage
    # Each image: input_size[0] * input_size[1] * 3 * 4 bytes (float32)
    image_memory = input_size[0] * input_size[1] * 3 * 4
    batch_memory = image_memory * batch_size
    model_memory = batch_memory * 2  # Rough estimate for model weights

    total_memory_mb = (batch_memory + model_memory) / (1024 * 1024)

    print(f"📊 Estimated memory per batch: {batch_memory / (1024*1024):.1f} MB")
    print(f"📊 Estimated total memory: {total_memory_mb:.1f} MB")

    # Check available system memory
    try:
        import psutil

        available_memory = psutil.virtual_memory().available / (
            1024 * 1024 * 1024
        )  # GB
        print(f"📊 Available system memory: {available_memory:.1f} GB")

        if total_memory_mb > available_memory * 1024 * 0.8:  # 80% of available memory
            print(
                "⚠️ Warning: High memory usage expected. Consider reducing batch size."
            )
            return False
        else:
            print("✅ Memory requirements look reasonable.")
            return True
    except ImportError:
        print("ℹ️ psutil not available, skipping system memory check.")
        return True


def validate_windows_safe_settings():
    """
    Validate that all settings are Windows-safe
    """
    print(f"\n✅ Validating Windows-Safe Settings...")

    # Define recommended settings
    recommended_settings = {
        "batch_size": {"min": 2, "max": 16, "recommended": 8},
        "input_size": {"min": (64, 64), "max": (224, 224), "recommended": (224, 224)},
        "workers": {"value": 0, "description": "No multiprocessing"},
        "use_multiprocessing": {
            "value": False,
            "description": "Multiprocessing disabled",
        },
        "max_queue_size": {"min": 5, "max": 15, "recommended": 10},
    }

    print("📋 Recommended Windows-Safe Settings:")
    for setting, values in recommended_settings.items():
        if setting in ["workers", "use_multiprocessing"]:
            print(f"  {setting}: {values['value']} ({values['description']})")
        elif setting == "max_queue_size":
            print(
                f"  {setting}: {values['recommended']} (range: {values['min']}-{values['max']})"
            )
        else:
            print(
                f"  {setting}: {values['recommended']} (range: {values['min']}-{values['max']})"
            )

    return True


def generate_windows_safe_config():
    """
    Generate a Windows-safe configuration template
    """
    print(f"\n📝 Generating Windows-Safe Configuration Template...")

    config_template = """
# Windows-Safe TensorFlow Configuration
import os
import tensorflow as tf

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Windows-specific optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Configure GPU memory growth (if using GPU)
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
except Exception as e:
    print(f"GPU configuration failed: {e}")

# Set thread configuration for single-threaded operation
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Windows-Safe Training Configuration
BATCH_SIZE = 8  # Reduced for Windows stability
INPUT_SIZE = (224, 224)
EPOCHS = 30
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

# Windows-Safe Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Reduced for stability
    width_shift_range=0.1,  # Reduced for stability
    height_shift_range=0.1,  # Reduced for stability
    shear_range=0.1,  # Reduced for stability
    zoom_range=0.1,  # Reduced for stability
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Windows-Safe Generator Creation
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    workers=0,  # CRITICAL: No multiprocessing
    use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
    max_queue_size=10  # Reduced queue size
)

# Windows-Safe Model Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1,
    workers=0,  # CRITICAL: No multiprocessing
    use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
    max_queue_size=10,  # Reduced queue size
    shuffle=True
)

# Windows-Safe Model Prediction
predictions = model.predict(
    val_generator,
    workers=0,  # CRITICAL: No multiprocessing
    use_multiprocessing=False,  # CRITICAL: Disable multiprocessing
    verbose=1
)
"""

    # Save configuration template
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = f"windows_safe_config_{timestamp}.py"

    with open(config_file, "w") as f:
        f.write(config_template)

    print(f"✅ Configuration template saved as: {config_file}")
    return config_file


def main():
    """
    Main function to run all checks
    """
    print("=" * 80)
    print("🔍 Windows TensorFlow Configuration Checker")
    print("=" * 80)
    print("Validating settings for Windows-safe training...")
    print("=" * 80)

    # Configuration
    DATASET_PATH = r"E:\rudra\project\dataset"
    BATCH_SIZE = 8
    INPUT_SIZE = (224, 224)

    # Run all checks
    checks = []

    # Check 1: TensorFlow configuration
    checks.append(check_tensorflow_configuration())

    # Check 2: Dataset structure
    checks.append(check_dataset_structure(DATASET_PATH))

    # Check 3: Data generator configuration
    checks.append(
        test_data_generator_configuration(DATASET_PATH, BATCH_SIZE, INPUT_SIZE)
    )

    # Check 4: Memory requirements
    checks.append(check_memory_requirements(BATCH_SIZE, INPUT_SIZE))

    # Check 5: Windows-safe settings validation
    checks.append(validate_windows_safe_settings())

    # Generate configuration template
    config_file = generate_windows_safe_config()

    # Summary
    print("\n" + "=" * 80)
    print("📊 Configuration Check Summary")
    print("=" * 80)

    passed_checks = sum(checks)
    total_checks = len(checks)

    print(f"✅ Passed checks: {passed_checks}/{total_checks}")

    if passed_checks == total_checks:
        print("🎉 All checks passed! Your configuration is Windows-safe.")
        print("\n🚀 You can now proceed with training using:")
        print(f"   python windows_training_guide.py")
        print(f"\n📝 Or use the generated configuration template:")
        print(f"   {config_file}")
    else:
        print("⚠️ Some checks failed. Please review the issues above.")
        print("\n🔧 Recommended actions:")
        print("1. Fix dataset structure if needed")
        print("2. Reduce batch size if memory issues detected")
        print("3. Ensure all multiprocessing is disabled")
        print("4. Use the generated configuration template")

    print("\n" + "=" * 80)
    print("💡 Tips for Windows Training:")
    print("• Always set workers=0 and use_multiprocessing=False")
    print("• Use small batch sizes (≤ 16)")
    print("• Monitor memory usage during training")
    print("• Close other applications to free memory")
    print("• Restart Python if issues persist")
    print("=" * 80)


if __name__ == "__main__":
    main()
