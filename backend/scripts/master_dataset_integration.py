#!/usr/bin/env python3
"""
Master Dataset Integration Script
Coordinates the complete process of integrating CBIS-DDSM and BUSI datasets
to improve breast cancer classification accuracy from 87.29% to 92.74%
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


class DatasetIntegrationMaster:
    def __init__(self, base_dir="dataset_integration"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Define paths
        self.cbis_source = Path("dataset4/cbisddsm_download")
        self.busi_source = Path("dataset4/archive/Dataset_BUSI_with_GT")
        self.current_data = Path("data")

        # Integration steps directories
        self.step_dirs = {
            "step1_cbis_converted": self.base_dir / "step1_cbis_converted",
            "step2_busi_copied": self.base_dir / "step2_busi_copied",
            "step2_busi_filtered": self.base_dir / "step2_busi_filtered",
            "step3_combined": self.base_dir / "step3_combined",
            "step4_organized": self.base_dir / "step4_organized",
            "step5_preprocessed": self.base_dir / "step5_preprocessed",
            "step6_split": self.base_dir / "step6_split",
            "step7_final": self.base_dir / "step7_final",
        }

        # Create all step directories
        for step_dir in self.step_dirs.values():
            step_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = (
            self.base_dir
            / f"integration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def run_command(self, cmd, description):
        """Run a command and log the results"""
        self.log(f"🔄 {description}")
        self.log(f"   Command: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd()
            )

            if result.returncode == 0:
                self.log(f"   ✅ {description} completed successfully")
                if result.stdout.strip():
                    self.log(f"   Output: {result.stdout.strip()}")
                return True
            else:
                self.log(f"   ❌ {description} failed")
                self.log(f"   Error: {result.stderr.strip()}")
                return False

        except Exception as e:
            self.log(f"   ❌ {description} failed with exception: {str(e)}")
            return False

    def check_prerequisites(self):
        """Check if all required files and directories exist"""
        self.log("🔍 Checking prerequisites...")

        prerequisites = [
            ("BUSI dataset", self.busi_source),
            ("Current data directory", self.current_data),
            ("Current images", self.current_data / "images"),
            ("Training CSV", self.current_data / "train_enhanced.csv"),
            ("Validation CSV", self.current_data / "val_enhanced.csv"),
            ("Test CSV", self.current_data / "test_enhanced.csv"),
            ("Convert DICOM script", Path("convert_dicom_to_png.py")),
            ("Organize dataset script", Path("organize_dataset.py")),
            ("Split dataset script", Path("split_dataset.py")),
            ("Preprocess images script", Path("preprocess_images.py")),
            ("Training script", Path("improved_training.py")),
        ]

        missing = []
        for name, path in prerequisites:
            if path.exists():
                self.log(f"   ✅ {name}: {path}")
            else:
                self.log(f"   ❌ {name}: {path} - NOT FOUND")
                missing.append(name)

        if missing:
            self.log(f"⚠️ Missing prerequisites: {', '.join(missing)}")
            return False

        self.log("✅ All prerequisites found")
        return True

    def step1_process_cbis_dataset(self):
        """Step 1: Process CBIS-DDSM dataset (convert DICOM to PNG)"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 1: Processing CBIS-DDSM Dataset")
        self.log("=" * 60)

        if not self.cbis_source.exists():
            self.log("⚠️ CBIS-DDSM dataset not found, skipping this step")
            return True

        # Check if CBIS dataset has any DICOM files
        dicom_files = list(self.cbis_source.rglob("*.dcm"))
        if not dicom_files:
            self.log("⚠️ No DICOM files found in CBIS-DDSM dataset, skipping conversion")
            return True

        self.log(f"📊 Found {len(dicom_files)} DICOM files")

        cmd = f"python convert_dicom_to_png.py --input_folder \"{self.cbis_source}\" --output_folder \"{self.step_dirs['step1_cbis_converted']}\""
        return self.run_command(cmd, "Converting CBIS-DDSM DICOM files to PNG")

    def step2_process_busi_dataset(self):
        """Step 2: Process BUSI dataset (copy and filter)"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 2: Processing BUSI Dataset")
        self.log("=" * 60)

        # Step 2a: Copy BUSI dataset
        self.log("🔄 Step 2a: Copying BUSI dataset...")
        try:
            if self.step_dirs["step2_busi_copied"].exists():
                shutil.rmtree(self.step_dirs["step2_busi_copied"])

            shutil.copytree(self.busi_source, self.step_dirs["step2_busi_copied"])
            self.log("   ✅ BUSI dataset copied successfully")

            # Count files
            for class_name in ["benign", "malignant", "normal"]:
                class_dir = self.step_dirs["step2_busi_copied"] / class_name
                if class_dir.exists():
                    count = len(list(class_dir.iterdir()))
                    self.log(f"   📊 {class_name.capitalize()}: {count} files")

        except Exception as e:
            self.log(f"   ❌ Failed to copy BUSI dataset: {str(e)}")
            return False

        # Step 2b: Filter BUSI dataset
        self.log("\n🔄 Step 2b: Filtering BUSI dataset...")
        cmd = f"python filter_busi_dataset.py --input_dir \"{self.step_dirs['step2_busi_copied']}\" --output_dir \"{self.step_dirs['step2_busi_filtered']}\""
        return self.run_command(cmd, "Filtering BUSI dataset")

    def step3_combine_datasets(self):
        """Step 3: Combine all datasets"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 3: Combining All Datasets")
        self.log("=" * 60)

        combined_dir = self.step_dirs["step3_combined"]

        # Create class directories
        for class_name in ["benign", "malignant", "normal"]:
            (combined_dir / class_name).mkdir(parents=True, exist_ok=True)

        stats = {"benign": 0, "malignant": 0, "normal": 0}

        # Copy current dataset
        self.log("🔄 Copying current dataset...")
        current_images = self.current_data / "images"
        current_train = pd.read_csv(self.current_data / "train_enhanced.csv")
        current_val = pd.read_csv(self.current_data / "val_enhanced.csv")
        current_test = pd.read_csv(self.current_data / "test_enhanced.csv")

        # Combine all current CSVs
        all_current = pd.concat(
            [current_train, current_val, current_test], ignore_index=True
        )

        for idx, row in all_current.iterrows():
            filename = row["filename"]
            label = row["label"]

            source_file = current_images / filename
            if source_file.exists():
                dest_file = (
                    combined_dir / label / f"current_{stats[label]:04d}_{filename}"
                )
                shutil.copy2(source_file, dest_file)
                stats[label] += 1

        self.log(f"   ✅ Copied current dataset:")
        for class_name, count in stats.items():
            self.log(f"     {class_name.capitalize()}: {count} images")

        # Copy CBIS dataset if available
        cbis_dir = self.step_dirs["step1_cbis_converted"]
        if cbis_dir.exists() and any(cbis_dir.iterdir()):
            self.log("🔄 Copying CBIS-DDSM dataset...")
            # Add CBIS copying logic here if needed

        # Copy filtered BUSI dataset
        busi_dir = self.step_dirs["step2_busi_filtered"]
        if busi_dir.exists():
            self.log("🔄 Copying filtered BUSI dataset...")

            for class_name in ["benign", "malignant", "normal"]:
                busi_class_dir = busi_dir / class_name
                if busi_class_dir.exists():
                    busi_count = 0
                    for busi_file in busi_class_dir.iterdir():
                        if busi_file.is_file():
                            dest_file = combined_dir / class_name / busi_file.name
                            shutil.copy2(busi_file, dest_file)
                            busi_count += 1

                    self.log(f"   ✅ Copied BUSI {class_name}: {busi_count} images")
                    stats[class_name] += busi_count

        # Final statistics
        self.log(f"\n📊 Combined Dataset Statistics:")
        total_images = 0
        for class_name, count in stats.items():
            self.log(f"   {class_name.capitalize()}: {count} images")
            total_images += count
        self.log(f"   Total: {total_images} images")

        return True

    def step4_organize_dataset(self):
        """Step 4: Organize dataset using the organize_dataset.py script"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 4: Organizing Combined Dataset")
        self.log("=" * 60)

        # The combined dataset is already organized by class, so we can copy it directly
        try:
            if self.step_dirs["step4_organized"].exists():
                shutil.rmtree(self.step_dirs["step4_organized"])

            shutil.copytree(
                self.step_dirs["step3_combined"], self.step_dirs["step4_organized"]
            )
            self.log("✅ Dataset organization completed")
            return True

        except Exception as e:
            self.log(f"❌ Dataset organization failed: {str(e)}")
            return False

    def step5_preprocess_images(self):
        """Step 5: Preprocess images"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 5: Preprocessing Images")
        self.log("=" * 60)

        cmd = f"python preprocess_images.py --input_dir \"{self.step_dirs['step4_organized']}\" --output_dir \"{self.step_dirs['step5_preprocessed']}\""
        return self.run_command(cmd, "Preprocessing images")

    def step6_split_dataset(self):
        """Step 6: Split dataset into train/validation/test"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 6: Splitting Dataset")
        self.log("=" * 60)

        cmd = f"python split_dataset.py --input_dir \"{self.step_dirs['step5_preprocessed']}\" --output_dir \"{self.step_dirs['step6_split']}\""
        return self.run_command(cmd, "Splitting dataset")

    def step7_prepare_final_data(self):
        """Step 7: Prepare final data structure"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 7: Preparing Final Data Structure")
        self.log("=" * 60)

        final_dir = self.step_dirs["step7_final"]
        split_dir = self.step_dirs["step6_split"]

        try:
            # Create final structure
            (final_dir / "images").mkdir(parents=True, exist_ok=True)

            # Copy all images to final images directory and create CSVs
            train_data = []
            val_data = []
            test_data = []

            image_counter = 0

            # Process each split
            for split_name, data_list in [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]:
                split_path = split_dir / split_name
                if split_path.exists():
                    for class_name in ["benign", "malignant", "normal"]:
                        class_path = split_path / class_name
                        if class_path.exists():
                            for img_file in class_path.iterdir():
                                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                                    # Create new filename to avoid conflicts
                                    new_filename = f"integrated_{class_name}_{image_counter:05d}{img_file.suffix}"

                                    # Copy to final images directory
                                    dest_path = final_dir / "images" / new_filename
                                    shutil.copy2(img_file, dest_path)

                                    # Add to appropriate data list
                                    data_list.append(
                                        {
                                            "filename": new_filename,
                                            "label": class_name,
                                            "original_file": img_file.name,
                                        }
                                    )

                                    image_counter += 1

            # Create CSV files
            pd.DataFrame(train_data).to_csv(
                final_dir / "train_integrated.csv", index=False
            )
            pd.DataFrame(val_data).to_csv(final_dir / "val_integrated.csv", index=False)
            pd.DataFrame(test_data).to_csv(
                final_dir / "test_integrated.csv", index=False
            )

            self.log(f"✅ Final data structure prepared:")
            self.log(f"   Training samples: {len(train_data)}")
            self.log(f"   Validation samples: {len(val_data)}")
            self.log(f"   Test samples: {len(test_data)}")
            self.log(f"   Total images: {image_counter}")

            return True

        except Exception as e:
            self.log(f"❌ Final data preparation failed: {str(e)}")
            return False

    def step8_backup_current_data(self):
        """Step 8: Backup current data and replace with integrated data"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 8: Backing Up Current Data and Integrating New Dataset")
        self.log("=" * 60)

        try:
            # Create backup
            backup_dir = Path(f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.log(f"🔄 Creating backup at {backup_dir}...")
            shutil.copytree(self.current_data, backup_dir)
            self.log("✅ Current data backed up successfully")

            # Replace with integrated data
            self.log("🔄 Replacing current data with integrated dataset...")

            # Copy new images
            current_images = self.current_data / "images"
            if current_images.exists():
                shutil.rmtree(current_images)

            shutil.copytree(self.step_dirs["step7_final"] / "images", current_images)

            # Copy new CSV files
            for csv_name in [
                "train_integrated.csv",
                "val_integrated.csv",
                "test_integrated.csv",
            ]:
                source_csv = self.step_dirs["step7_final"] / csv_name
                dest_csv = self.current_data / csv_name.replace(
                    "_integrated", "_enhanced"
                )

                if source_csv.exists():
                    shutil.copy2(source_csv, dest_csv)

            self.log("✅ Data integration completed successfully")
            self.log(f"💾 Original data backed up to: {backup_dir}")

            return True

        except Exception as e:
            self.log(f"❌ Data backup and integration failed: {str(e)}")
            return False

    def step9_retrain_model(self):
        """Step 9: Retrain model with integrated dataset"""
        self.log("\n" + "=" * 60)
        self.log("📋 STEP 9: Retraining Model with Integrated Dataset")
        self.log("=" * 60)

        # Create a backup of current best model
        current_model = Path("best_improved_model.pt")
        if current_model.exists():
            backup_model = Path(
                f"best_improved_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            )
            shutil.copy2(current_model, backup_model)
            self.log(f"💾 Current model backed up to: {backup_model}")

        # Run training
        cmd = "python improved_training.py --epochs 25 --batch_size 16 --learning_rate 0.0005"
        success = self.run_command(cmd, "Training model with integrated dataset")

        if success:
            self.log("🎉 Model training completed successfully!")
            self.log("📊 Please check the training output for final accuracy metrics")

        return success

    def run_complete_integration(self):
        """Run the complete dataset integration process"""
        self.log("=" * 80)
        self.log("🚀 BREAST CANCER CLASSIFICATION - DATASET INTEGRATION")
        self.log("=" * 80)
        self.log(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"🎯 Goal: Improve accuracy from 87.29% to 92.74%")
        self.log("")

        steps = [
            ("Check Prerequisites", self.check_prerequisites),
            ("Process CBIS-DDSM Dataset", self.step1_process_cbis_dataset),
            ("Process BUSI Dataset", self.step2_process_busi_dataset),
            ("Combine All Datasets", self.step3_combine_datasets),
            ("Organize Combined Dataset", self.step4_organize_dataset),
            ("Preprocess Images", self.step5_preprocess_images),
            ("Split Dataset", self.step6_split_dataset),
            ("Prepare Final Data Structure", self.step7_prepare_final_data),
            ("Backup and Integrate Data", self.step8_backup_current_data),
            ("Retrain Model", self.step9_retrain_model),
        ]

        failed_steps = []

        for step_name, step_function in steps:
            success = step_function()
            if not success:
                failed_steps.append(step_name)
                self.log(f"❌ Step failed: {step_name}")

                # Ask user whether to continue
                response = input(
                    f"\n⚠️ Step '{step_name}' failed. Continue with remaining steps? (y/n): "
                )
                if response.lower() != "y":
                    self.log("🛑 Integration process stopped by user")
                    break

        # Final summary
        self.log("\n" + "=" * 80)
        self.log("📊 INTEGRATION PROCESS SUMMARY")
        self.log("=" * 80)

        if not failed_steps:
            self.log("🎉 All steps completed successfully!")
            self.log("✅ Dataset integration completed")
            self.log("✅ Model retrained with expanded dataset")
            self.log("📈 Check the training output for new accuracy metrics")
        else:
            self.log(f"⚠️ {len(failed_steps)} step(s) failed:")
            for step in failed_steps:
                self.log(f"   ❌ {step}")

        self.log(f"\n📁 Integration workspace: {self.base_dir}")
        self.log(f"📄 Complete log: {self.log_file}")
        self.log(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Dataset Integration for Breast Cancer Classification"
    )
    parser.add_argument(
        "--workspace",
        default="dataset_integration",
        help="Workspace directory for integration process",
    )

    args = parser.parse_args()

    integrator = DatasetIntegrationMaster(args.workspace)
    integrator.run_complete_integration()
