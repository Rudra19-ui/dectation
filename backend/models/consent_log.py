"""Consent logging module for medical image analysis application."""

import os
import json
import datetime
import uuid
from pathlib import Path

# Define the log directory
LOG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs" / "consent"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def log_consent(consent_given, user_ip, request_id=None):
    """
    Log user consent for medical image processing.
    
    Args:
        consent_given (bool): Whether consent was given
        user_ip (str): Anonymized user IP address
        request_id (str, optional): Unique request identifier. Defaults to None.
    
    Returns:
        str: The log entry ID
    """
    # Generate a unique ID for this consent log
    log_id = str(uuid.uuid4())
    
    # Use provided request_id or generate a new one
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # Create log entry
    log_entry = {
        "log_id": log_id,
        "request_id": request_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "consent_given": consent_given,
        "user_ip_hash": hash(user_ip),  # Store hash of IP for privacy
        "consent_version": "1.0"
    }
    
    # Write to log file
    log_file = LOG_DIR / f"{log_id}.json"
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    print(f"Consent logged: {log_id} (Consent given: {consent_given})")
    return log_id


def get_consent_status(request_id):
    """
    Check if consent was given for a specific request.
    
    Args:
        request_id (str): The request identifier
    
    Returns:
        bool: True if consent was given, False otherwise
    """
    # Search through log files for the request_id
    for log_file in LOG_DIR.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                log_entry = json.load(f)
                if log_entry.get("request_id") == request_id:
                    return log_entry.get("consent_given", False)
        except (json.JSONDecodeError, IOError):
            continue
    
    return False


def revoke_consent(request_id):
    """
    Revoke previously given consent.
    
    Args:
        request_id (str): The request identifier
    
    Returns:
        bool: True if consent was successfully revoked, False otherwise
    """
    # Search through log files for the request_id
    for log_file in LOG_DIR.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                log_entry = json.load(f)
                
            if log_entry.get("request_id") == request_id:
                # Update the consent status
                log_entry["consent_given"] = False
                log_entry["revoked_at"] = datetime.datetime.now().isoformat()
                
                # Write the updated entry back to the file
                with open(log_file, 'w') as f:
                    json.dump(log_entry, f, indent=2)
                
                print(f"Consent revoked for request: {request_id}")
                return True
        except (json.JSONDecodeError, IOError):
            continue
    
    return False