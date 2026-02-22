#!/usr/bin/env python3
"""
DICOM Anonymization Utilities

This module provides functions for anonymizing DICOM files by removing PHI (Protected Health Information).
"""

import io
import logging
from typing import Dict, List, Optional, Union, Tuple

import pydicom
from pydicom import Dataset
from pydicom.uid import generate_uid

# Configure logging
logger = logging.getLogger(__name__)

# PHI tags that should be removed or anonymized
PHI_TAGS = [
    # Patient information
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephone",
    "PatientMotherBirthName",
    "PatientBirthName",
    "PatientReligiousPreference",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    "PatientComments",
    
    # Institution information
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephone",
    "PhysiciansOfRecord",
    "OperatorsName",
    "PerformingPhysicianName",
    "RequestAttributesSequence",
    
    # Study information that could identify patient
    "AccessionNumber",
    "StudyID",
    "RequestingService",
    "CurrentPatientLocation",
    "RequestingPhysician",
]

# Tags that should be given new UIDs
UID_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "MediaStorageSOPInstanceUID",
]


def anonymize_dicom_bytes(data: bytes, keep_uids: bool = False) -> bytes:
    """
    Anonymize DICOM data by removing PHI tags.
    
    Args:
        data: DICOM data as bytes
        keep_uids: If True, preserve original UIDs (useful for linking anonymized data)
        
    Returns:
        Anonymized DICOM data as bytes
    """
    try:
        # Read DICOM dataset
        ds = pydicom.dcmread(io.BytesIO(data))
        
        # Anonymize dataset
        ds = anonymize_dataset(ds, keep_uids=keep_uids)
        
        # Save to bytes
        buffer = io.BytesIO()
        ds.save_as(buffer)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error anonymizing DICOM: {e}")
        raise


def anonymize_dicom(ds: Dataset, keep_uids: bool = False) -> Dataset:
    """
    Anonymize a DICOM dataset by removing PHI tags.
    
    Args:
        ds: DICOM dataset to anonymize
        keep_uids: If True, preserve original UIDs
        
    Returns:
        Anonymized DICOM dataset
    """
    return anonymize_dataset(ds, keep_uids=keep_uids)


def anonymize_dataset(ds: Dataset, keep_uids: bool = False) -> Dataset:
    """
    Anonymize a DICOM dataset by removing PHI tags.
    
    Args:
        ds: DICOM dataset
        keep_uids: If True, preserve original UIDs
        
    Returns:
        Anonymized DICOM dataset
    """
    # Remove PHI tags
    for tag in PHI_TAGS:
        if tag in ds:
            delattr(ds, tag)
    
    # Generate new UIDs if needed
    if not keep_uids:
        uid_map = {}
        for tag in UID_TAGS:
            if tag in ds:
                old_uid = getattr(ds, tag)
                if old_uid not in uid_map:
                    uid_map[old_uid] = generate_uid()
                setattr(ds, tag, uid_map[old_uid])
                
        # Update file meta UIDs if present
        if hasattr(ds, 'file_meta'):
            if hasattr(ds.file_meta, 'MediaStorageSOPInstanceUID'):
                if ds.SOPInstanceUID in uid_map:
                    ds.file_meta.MediaStorageSOPInstanceUID = uid_map[ds.SOPInstanceUID]
    
    # Add anonymization indicators
    ds.PatientIdentityRemoved = "YES"
    
    return ds


def get_phi_tags_in_dataset(ds: Dataset) -> List[str]:
    """
    Get a list of PHI tags present in the dataset.
    
    Args:
        ds: DICOM dataset
        
    Returns:
        List of PHI tag names present in the dataset
    """
    return [tag for tag in PHI_TAGS if tag in ds]


def has_phi(ds: Dataset) -> bool:
    """
    Check if the dataset contains any PHI tags.
    
    Args:
        ds: DICOM dataset
        
    Returns:
        True if PHI tags are present, False otherwise
    """
    return len(get_phi_tags_in_dataset(ds)) > 0


def anonymize_dicom_file(input_path: str, output_path: str, keep_uids: bool = False) -> bool:
    """
    Anonymize a DICOM file and save to a new file.
    
    Args:
        input_path: Path to input DICOM file
        output_path: Path to output anonymized DICOM file
        keep_uids: If True, preserve original UIDs
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(input_path)
        
        # Anonymize dataset
        ds = anonymize_dataset(ds, keep_uids=keep_uids)
        
        # Save anonymized dataset
        ds.save_as(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Error anonymizing DICOM file: {e}")
        return False