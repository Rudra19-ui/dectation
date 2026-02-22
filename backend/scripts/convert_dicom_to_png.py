#!/usr/bin/env python3
"""
DICOM to PNG Converter for Mammogram Images
Converts all .dcm files in a folder to PNG format using pydicom and OpenCV
Optimized for mammogram images with proper windowing and normalization
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.errors import InvalidDicomError

# Suppress warnings
warnings.filterwarnings("ignore")


class DicomToPngConverter:
    """
    DICOM to PNG converter with mammogram-specific optimizations
    """

    def __init__(
        self, input_folder, output_folder=None, window_center=None, window_width=None
    ):
        """
        Initialize the converter

        Args:
            input_folder (str): Path to folder containing DICOM files
            output_folder (str): Path to output folder (optional)
            window_center (int): Window center for mammogram display (optional)
            window_width (int): Window width for mammogram display (optional)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = (
            Path(output_folder)
            if output_folder
            else self.input_folder / "converted_png"
        )
        self.window_center = window_center
        self.window_width = window_width

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_files": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "skipped_files": 0,
            "errors": [],
        }

    def find_dicom_files(self):
        """
        Find all DICOM files in the input folder and subfolders

        Returns:
            list: List of DICOM file paths
        """
        dicom_files = []

        # Search for .dcm files recursively
        for file_path in self.input_folder.rglob("*.dcm"):
            dicom_files.append(file_path)

        # Also search for files without .dcm extension that might be DICOM
        for file_path in self.input_folder.rglob("*"):
            if file_path.is_file() and not file_path.suffix.lower() in [
                ".dcm",
                ".png",
                ".jpg",
                ".jpeg",
            ]:
                try:
                    # Try to read as DICOM
                    dcmread(str(file_path), stop_before_pixels=True)
                    dicom_files.append(file_path)
                except (InvalidDicomError, Exception):
                    pass

        return dicom_files

    def get_dicom_info(self, dicom_file):
        """
        Extract basic information from DICOM file

        Args:
            dicom_file (pydicom.Dataset): DICOM dataset

        Returns:
            dict: Dictionary containing DICOM information
        """
        info = {}

        try:
            # Basic patient information
            info["patient_name"] = str(getattr(dicom_file, "PatientName", "Unknown"))
            info["patient_id"] = str(getattr(dicom_file, "PatientID", "Unknown"))
            info["study_date"] = str(getattr(dicom_file, "StudyDate", "Unknown"))

            # Image information
            info["modality"] = str(getattr(dicom_file, "Modality", "Unknown"))
            info["image_size"] = (
                getattr(dicom_file, "Rows", 0),
                getattr(dicom_file, "Columns", 0),
            )
            info["bits_allocated"] = getattr(dicom_file, "BitsAllocated", 16)
            info["samples_per_pixel"] = getattr(dicom_file, "SamplesPerPixel", 1)

            # Window settings
            info["window_center"] = getattr(dicom_file, "WindowCenter", None)
            info["window_width"] = getattr(dicom_file, "WindowWidth", None)

            # Manufacturer information
            info["manufacturer"] = str(getattr(dicom_file, "Manufacturer", "Unknown"))
            info["manufacturer_model"] = str(
                getattr(dicom_file, "ManufacturerModelName", "Unknown")
            )

        except Exception as e:
            info["error"] = str(e)

        return info

    def apply_window_level(self, pixel_array, window_center=None, window_width=None):
        """
        Apply window/level transformation to DICOM pixel data

        Args:
            pixel_array (numpy.ndarray): DICOM pixel array
            window_center (int): Window center value
            window_width (int): Window width value

        Returns:
            numpy.ndarray: Windowed pixel array
        """
        if window_center is None or window_width is None:
            # Use default mammogram window settings
            window_center = 1500
            window_width = 3000

        # Calculate window limits
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2

        # Apply windowing
        windowed = np.clip(pixel_array, window_min, window_max)

        # Normalize to 0-255 range
        normalized = ((windowed - window_min) / (window_max - window_min) * 255).astype(
            np.uint8
        )

        return normalized

    def enhance_mammogram(self, image):
        """
        Apply mammogram-specific image enhancement

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Enhanced image
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Apply unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        return enhanced

    def convert_single_file(self, dicom_path):
        """
        Convert a single DICOM file to PNG

        Args:
            dicom_path (Path): Path to DICOM file

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Read DICOM file
            dicom_file = dcmread(str(dicom_path))

            # Get pixel array
            pixel_array = dicom_file.pixel_array

            # Get DICOM information
            dicom_info = self.get_dicom_info(dicom_file)

            # Apply window/level transformation
            if self.window_center is not None and self.window_width is not None:
                # Use provided window settings
                windowed_image = self.apply_window_level(
                    pixel_array, self.window_center, self.window_width
                )
            elif dicom_info.get("window_center") and dicom_info.get("window_width"):
                # Use DICOM window settings
                windowed_image = self.apply_window_level(
                    pixel_array, dicom_info["window_center"], dicom_info["window_width"]
                )
            else:
                # Use default mammogram window settings
                windowed_image = self.apply_window_level(pixel_array)

            # Apply mammogram-specific enhancement
            enhanced_image = self.enhance_mammogram(windowed_image)

            # Generate output filename
            output_filename = dicom_path.stem + ".png"
            output_path = self.output_folder / output_filename

            # Save as PNG
            success = cv2.imwrite(str(output_path), enhanced_image)

            if success:
                print(f"✅ Converted: {dicom_path.name} -> {output_filename}")
                print(
                    f"   Size: {pixel_array.shape}, Modality: {dicom_info.get('modality', 'Unknown')}"
                )
                return True
            else:
                print(f"❌ Failed to save: {dicom_path.name}")
                self.stats["errors"].append(f"Failed to save {dicom_path.name}")
                return False

        except InvalidDicomError:
            print(f"⚠️ Skipped (not a valid DICOM): {dicom_path.name}")
            self.stats["skipped_files"] += 1
            return False
        except Exception as e:
            print(f"❌ Error converting {dicom_path.name}: {str(e)}")
            self.stats["errors"].append(f"Error converting {dicom_path.name}: {str(e)}")
            return False

    def convert_batch(self, max_files=None, show_progress=True):
        """
        Convert all DICOM files in batch

        Args:
            max_files (int): Maximum number of files to convert (for testing)
            show_progress (bool): Show progress information
        """
        # Find DICOM files
        dicom_files = self.find_dicom_files()
        self.stats["total_files"] = len(dicom_files)

        if show_progress:
            print(f"🔍 Found {len(dicom_files)} DICOM files")
            print(f"📁 Input folder: {self.input_folder}")
            print(f"📁 Output folder: {self.output_folder}")
            print("=" * 60)

        if max_files:
            dicom_files = dicom_files[:max_files]
            print(f"⚠️ Limiting conversion to {max_files} files (testing mode)")

        # Convert files
        for i, dicom_path in enumerate(dicom_files, 1):
            if show_progress:
                print(f"\n📄 Processing {i}/{len(dicom_files)}: {dicom_path.name}")

            success = self.convert_single_file(dicom_path)

            if success:
                self.stats["successful_conversions"] += 1
            else:
                self.stats["failed_conversions"] += 1

        # Print summary
        self.print_summary()

    def print_summary(self):
        """
        Print conversion summary
        """
        print("\n" + "=" * 60)
        print("📊 CONVERSION SUMMARY")
        print("=" * 60)
        print(f"📁 Input folder: {self.input_folder}")
        print(f"📁 Output folder: {self.output_folder}")
        print(f"📄 Total files found: {self.stats['total_files']}")
        print(f"✅ Successful conversions: {self.stats['successful_conversions']}")
        print(f"❌ Failed conversions: {self.stats['failed_conversions']}")
        print(f"⚠️ Skipped files: {self.stats['skipped_files']}")

        if self.stats["errors"]:
            print(f"\n❌ Errors encountered:")
            for error in self.stats["errors"][:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.stats["errors"]) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more errors")

        success_rate = (
            self.stats["successful_conversions"] / max(self.stats["total_files"], 1)
        ) * 100
        print(f"\n📈 Success rate: {success_rate:.1f}%")

        if self.stats["successful_conversions"] > 0:
            print(f"🎉 Conversion completed successfully!")
            print(f"📁 Check output folder: {self.output_folder}")
        else:
            print(f"❌ No files were converted successfully.")


def create_sample_dicom_info(dicom_path):
    """
    Create a sample DICOM information display

    Args:
        dicom_path (Path): Path to DICOM file
    """
    try:
        dicom_file = dcmread(str(dicom_path))
        converter = DicomToPngConverter(".")
        info = converter.get_dicom_info(dicom_file)

        print(f"\n📋 DICOM Information for: {dicom_path.name}")
        print("-" * 40)
        for key, value in info.items():
            print(f"{key}: {value}")

        # Show pixel array info
        pixel_array = dicom_file.pixel_array
        print(f"\n📊 Pixel Array Information:")
        print(f"Shape: {pixel_array.shape}")
        print(f"Data type: {pixel_array.dtype}")
        print(f"Min value: {pixel_array.min()}")
        print(f"Max value: {pixel_array.max()}")
        print(f"Mean value: {pixel_array.mean():.2f}")

        return True
    except Exception as e:
        print(f"❌ Error reading DICOM file: {e}")
        return False


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Convert DICOM files to PNG format")
    parser.add_argument("input_folder", help="Input folder containing DICOM files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output folder for PNG files (default: input_folder/converted_png)",
    )
    parser.add_argument(
        "-wc", "--window-center", type=int, help="Window center for mammogram display"
    )
    parser.add_argument(
        "-ww", "--window-width", type=int, help="Window width for mammogram display"
    )
    parser.add_argument(
        "-m",
        "--max-files",
        type=int,
        help="Maximum number of files to convert (for testing)",
    )
    parser.add_argument("--info", help="Show DICOM information for a specific file")
    parser.add_argument(
        "--no-progress", action="store_true", help="Hide progress information"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🏥 DICOM to PNG Converter for Mammogram Images")
    print("=" * 60)

    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        print(f"❌ Input folder not found: {args.input_folder}")
        return

    # Show DICOM information if requested
    if args.info:
        if create_sample_dicom_info(Path(args.info)):
            return
        else:
            return

    # Create converter
    converter = DicomToPngConverter(
        input_folder=args.input_folder,
        output_folder=args.output,
        window_center=args.window_center,
        window_width=args.window_width,
    )

    # Convert files
    converter.convert_batch(
        max_files=args.max_files, show_progress=not args.no_progress
    )


if __name__ == "__main__":
    main()
