"""
Input validation and robust file handling utilities for breast cancer classification.
Validates uploads (file type, size, DICOM tags), rejects corrupt inputs with clear errors.
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple

import filetype
import numpy as np
import pydicom
from fastapi import HTTPException
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_IMAGE_DIMENSION = 4096  # Maximum width/height
MIN_IMAGE_DIMENSION = 32  # Minimum width/height
SUPPORTED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
}
SUPPORTED_DICOM_TYPES = {"application/dicom", "application/dcm"}


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_file_size(data: bytes) -> None:
    """
    Validate file size is within acceptable limits.

    Args:
        data: File data as bytes

    Raises:
        ValidationError: If file is too large
    """
    if len(data) > MAX_FILE_SIZE:
        raise ValidationError(
            f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )

    if len(data) < 100:  # Minimum file size
        raise ValidationError("File too small. Minimum size: 100 bytes")


def validate_bytes(data: bytes) -> bool:
    """
    Validate file bytes for safety and integrity.
    
    Args:
        data: File data as bytes
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    # Check file size
    validate_file_size(data)
    
    # Try to detect file type
    kind = filetype.guess(data)
    if kind is None:
        raise ValidationError("Unable to detect file type")
        
    mime_type = kind.mime
    
    # Handle based on file type
    if mime_type in SUPPORTED_IMAGE_TYPES:
        # For images, validate using image validation
        _, _ = validate_image_data(data)
    elif mime_type in SUPPORTED_DICOM_TYPES:
        # For DICOM, validate using DICOM validation
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            # Basic DICOM validation - check for required tags
            if not hasattr(ds, 'SOPClassUID'):
                raise ValidationError("Invalid DICOM: Missing SOPClassUID")
        except Exception as e:
            raise ValidationError(f"Invalid DICOM file: {str(e)}")
    else:
        raise ValidationError(f"Unsupported file type: {mime_type}")
    
    return True

def validate_image_data(data: bytes) -> Tuple[Image.Image, str]:
    """
    Validate and load image data.

    Args:
        data: Image data as bytes

    Returns:
        Tuple of (PIL Image, detected MIME type)

    Raises:
        ValidationError: If image is invalid or unsupported
    """
    try:
        # Detect file type
        kind = filetype.guess(data)
        if kind is None:
            raise ValidationError("Unable to detect file type")

        mime_type = kind.mime
        if mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValidationError(f"Unsupported image type: {mime_type}")

        # Load image with PIL
        image: Image.Image = Image.open(io.BytesIO(data))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Validate dimensions
        width, height = image.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            raise ValidationError(
                f"Image too small. Minimum dimensions: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}"
            )

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValidationError(
                f"Image too large. Maximum dimensions: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}"
            )

        # Validate image data integrity
        try:
            # Try to convert to numpy array to check for corruption
            np.array(image)
        except Exception as e:
            raise ValidationError(f"Corrupted image data: {str(e)}")

        return image, mime_type

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Invalid image format: {str(e)}")


def validate_dicom_data(data: bytes) -> Tuple[pydicom.Dataset, str]:
    """
    Validate and load DICOM data.

    Args:
        data: DICOM data as bytes

    Returns:
        Tuple of (DICOM dataset, detected MIME type)

    Raises:
        ValidationError: If DICOM is invalid or corrupted
    """
    try:
        # Detect file type
        kind = filetype.guess(data)
        mime_type = kind.mime if kind else "application/dicom"

        # Load DICOM dataset
        ds = pydicom.dcmread(io.BytesIO(data), force=True)

        # Validate required DICOM tags
        required_tags = ["SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID"]
        missing_tags = [tag for tag in required_tags if not hasattr(ds, tag)]

        if missing_tags:
            logger.warning(f"Missing DICOM tags: {missing_tags}")

        # Check for patient information (for anonymization)
        phi_tags = ["PatientName", "PatientID", "PatientBirthDate", "PatientAddress"]
        phi_present = [tag for tag in phi_tags if hasattr(ds, tag)]

        if phi_present:
            logger.info(f"PHI detected in DICOM: {phi_present}")

            # Validate image data
            try:
                if hasattr(ds, "pixel_array"):
                    pixel_array = ds.pixel_array
                    if pixel_array.size == 0:
                        raise ValidationError("DICOM contains no pixel data")

                    # Check dimensions
                    if len(pixel_array.shape) not in [2, 3]:
                        raise ValidationError(
                            f"Invalid DICOM pixel array shape: {pixel_array.shape}"
                        )

                    # Check for reasonable dimensions
                    if (
                        pixel_array.shape[0] < MIN_IMAGE_DIMENSION
                        or pixel_array.shape[1] < MIN_IMAGE_DIMENSION
                    ):
                        raise ValidationError(
                            f"DICOM image too small: {pixel_array.shape}"
                        )

                    if (
                        pixel_array.shape[0] > MAX_IMAGE_DIMENSION
                        or pixel_array.shape[1] > MAX_IMAGE_DIMENSION
                    ):
                        raise ValidationError(
                            f"DICOM image too large: {pixel_array.shape}"
                        )
                else:
                    # Check if pixel data exists but can't be accessed
                    if hasattr(ds, "PixelData") and ds.PixelData:
                        # Has pixel data but can't access pixel_array (encoding issue)
                        # This is acceptable for basic validation
                        pass
                    else:
                        raise ValidationError("DICOM file contains no pixel data")
            except Exception as e:
                # If pixel array access fails due to encoding, but DICOM structure is valid,
                # we can still accept it for basic validation
                if "pixel_array" in str(e).lower() or "encoding" in str(e).lower():
                    logger.warning(f"DICOM pixel data access failed: {e}")
                    # Check if we have basic DICOM structure
                    if not (
                        hasattr(ds, "SOPInstanceUID")
                        and hasattr(ds, "StudyInstanceUID")
                    ):
                        raise ValidationError("Invalid DICOM structure")
                else:
                    raise ValidationError(f"Invalid DICOM pixel data: {str(e)}")

        return ds, mime_type

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Invalid DICOM format: {str(e)}")


def validate_upload_bytes(data: bytes) -> Dict[str, Any]:
    """
    Comprehensive validation of uploaded file bytes.

    Args:
        data: File data as bytes

    Returns:
        Dictionary with validation results and metadata

    Raises:
        ValidationError: If validation fails
    """
    # Basic size validation
    validate_file_size(data)

    # Detect file type
    kind = filetype.guess(data)
    if kind is None:
        raise ValidationError("Unable to detect file type")

    mime_type = kind.mime
    file_extension = kind.extension

    result = {
        "mime_type": mime_type,
        "file_extension": file_extension,
        "file_size": len(data),
        "is_dicom": mime_type in SUPPORTED_DICOM_TYPES,
        "is_image": mime_type in SUPPORTED_IMAGE_TYPES,
        "validation_passed": True,
    }

    try:
        if mime_type in SUPPORTED_DICOM_TYPES:
            # Validate DICOM
            ds, _ = validate_dicom_data(data)
            result.update(
                {
                    "dicom_dataset": ds,
                    "has_phi": any(
                        hasattr(ds, tag)
                        for tag in ["PatientName", "PatientID", "PatientBirthDate"]
                    ),
                    "study_uid": getattr(ds, "StudyInstanceUID", None),
                    "series_uid": getattr(ds, "SeriesInstanceUID", None),
                    "instance_uid": getattr(ds, "SOPInstanceUID", None),
                }
            )

        elif mime_type in SUPPORTED_IMAGE_TYPES:
            # Validate image
            image, _ = validate_image_data(data)
            result.update(
                {"image": image, "image_size": image.size, "image_mode": image.mode}
            )

        else:
            raise ValidationError(f"Unsupported file type: {mime_type}")

    except ValidationError as e:
        logger.error(f"Validation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        raise ValidationError(f"File validation failed: {str(e)}")

    return result


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    import os
    import re

    # Remove path components
    filename = os.path.basename(filename)

    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 100:
        name = name[:100]

    return name + ext


def get_file_metadata(data: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract metadata from file data.

    Args:
        data: File data as bytes
        filename: Optional filename

    Returns:
        Dictionary with file metadata
    """
    metadata = {
        "filename": sanitize_filename(filename) if filename else "unknown",
        "file_size": len(data),
        "timestamp": None,
    }

    try:
        validation_result = validate_upload_bytes(data)
        metadata.update(
            {
                "mime_type": validation_result["mime_type"],
                "file_extension": validation_result["file_extension"],
                "is_dicom": validation_result["is_dicom"],
                "is_image": validation_result["is_image"],
            }
        )

        if validation_result["is_dicom"] and "dicom_dataset" in validation_result:
            ds = validation_result["dicom_dataset"]
            metadata.update(
                {
                    "study_uid": getattr(ds, "StudyInstanceUID", None),
                    "series_uid": getattr(ds, "SeriesInstanceUID", None),
                    "instance_uid": getattr(ds, "SOPInstanceUID", None),
                    "modality": getattr(ds, "Modality", None),
                    "study_date": getattr(ds, "StudyDate", None),
                    "study_time": getattr(ds, "StudyTime", None),
                }
            )

        elif validation_result["is_image"] and "image" in validation_result:
            image = validation_result["image"]
            metadata.update({"image_size": image.size, "image_mode": image.mode})

    except ValidationError:
        # If validation fails, still return basic metadata
        pass

    return metadata


# HTTP Exception helpers for FastAPI
def validation_error_to_http_exception(e: ValidationError) -> HTTPException:
    """Convert ValidationError to HTTPException for FastAPI."""
    return HTTPException(status_code=400, detail=str(e))


def file_too_large_exception() -> HTTPException:
    """Create HTTPException for file too large."""
    return HTTPException(
        status_code=413,
        detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB",
    )


def unsupported_file_type_exception(mime_type: str) -> HTTPException:
    """Create HTTPException for unsupported file type."""
    return HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {mime_type}. Supported types: {', '.join(SUPPORTED_IMAGE_TYPES | SUPPORTED_DICOM_TYPES)}",
    )


def validate_dicom(file_obj: io.BytesIO) -> Dict[str, Any]:
    """Validate DICOM file.
    
    Args:
        file_obj: DICOM file as BytesIO object
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Try to load DICOM dataset
        ds = pydicom.dcmread(file_obj, force=True)
        
        # Check if it has pixel data
        has_pixel_data = hasattr(ds, "PixelData") and ds.PixelData
        
        # Check required tags
        required_tags = ["SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID"]
        missing_tags = [tag for tag in required_tags if not hasattr(ds, tag)]
        
        # Validation result
        result = {
            "is_valid": has_pixel_data and not missing_tags,
            "message": "",
            "dataset": ds
        }
        
        if not has_pixel_data:
            result["message"] = "DICOM file contains no pixel data"
        elif missing_tags:
            result["message"] = f"Missing required DICOM tags: {', '.join(missing_tags)}"
            
        return result
        
    except Exception as e:
        return {
            "is_valid": False,
            "message": f"Invalid DICOM file: {str(e)}",
            "dataset": None
        }


def validate_image(file_obj: io.BytesIO) -> Dict[str, Any]:
    """Validate image file.
    
    Args:
        file_obj: Image file as BytesIO object
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Try to load image with PIL
        image = Image.open(file_obj)
        
        # Check dimensions
        width, height = image.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return {
                "is_valid": False,
                "message": f"Image too small. Minimum dimensions: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}",
                "image": None
            }
            
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return {
                "is_valid": False,
                "message": f"Image too large. Maximum dimensions: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}",
                "image": None
            }
            
        # Validation passed
        return {
            "is_valid": True,
            "message": "",
            "image": image
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "message": f"Invalid image file: {str(e)}",
            "image": None
        }


def extract_dicom_image(file_obj: io.BytesIO) -> np.ndarray:
    """Extract image from DICOM file.
    
    Args:
        file_obj: DICOM file as BytesIO object
        
    Returns:
        Image as numpy array
    """
    ds = pydicom.dcmread(file_obj)
    
    # Get pixel array
    pixel_array = ds.pixel_array
    
    # Normalize to 0-255 if needed
    if pixel_array.max() > 255:
        pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
    
    return pixel_array


def extract_dicom_metadata(file_obj: io.BytesIO) -> Dict[str, Any]:
    """Extract metadata from DICOM file.
    
    Args:
        file_obj: DICOM file as BytesIO object
        
    Returns:
        Dictionary with DICOM metadata
    """
    ds = pydicom.dcmread(file_obj)
    
    # Extract common metadata
    metadata = {
        "study_uid": getattr(ds, "StudyInstanceUID", None),
        "series_uid": getattr(ds, "SeriesInstanceUID", None),
        "instance_uid": getattr(ds, "SOPInstanceUID", None),
        "modality": getattr(ds, "Modality", None),
        "study_date": getattr(ds, "StudyDate", None),
        "study_time": getattr(ds, "StudyTime", None),
    }
    
    return metadata
