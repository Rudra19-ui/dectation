import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os

class AuditLog:
    """
    A class to handle audit logging for clinical deployment.
    In a production environment, this would use a database.
    For this implementation, we'll use a simple file-based approach.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the audit log.
        
        Args:
            log_file: Path to the log file. If None, a default path will be used.
        """
        if log_file is None:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, 'audit_log.jsonl')
        
        self.log_file = log_file
    
    def log_action(self, 
                  user_id: str, 
                  action: str, 
                  resource_id: Optional[str] = None, 
                  model_version: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an action to the audit trail.
        
        Args:
            user_id: The ID of the user performing the action
            action: The action being performed (e.g., 'view', 'analyze', 'share')
            resource_id: The ID of the resource being acted upon (e.g., image_id, report_id)
            model_version: The version of the model used (if applicable)
            metadata: Additional metadata to include in the log entry
            
        Returns:
            The request_id of the log entry
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'request_id': request_id,
            'user_id': user_id,
            'action': action,
            'timestamp': timestamp,
            'resource_id': resource_id,
            'model_version': model_version
        }
        
        # Add additional metadata if provided
        if metadata:
            log_entry.update(metadata)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return request_id
    
    def get_logs(self, 
                user_id: Optional[str] = None, 
                action: Optional[str] = None,
                resource_id: Optional[str] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve logs from the audit trail with optional filtering.
        
        Args:
            user_id: Filter by user ID
            action: Filter by action type
            resource_id: Filter by resource ID
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of logs to return
            
        Returns:
            A list of log entries matching the filters
        """
        logs = []
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Apply filters
                        if user_id and log_entry.get('user_id') != user_id:
                            continue
                        if action and log_entry.get('action') != action:
                            continue
                        if resource_id and log_entry.get('resource_id') != resource_id:
                            continue
                        
                        # Time filters
                        if start_time or end_time:
                            log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                            if start_time and log_time < start_time:
                                continue
                            if end_time and log_time > end_time:
                                continue
                        
                        logs.append(log_entry)
                        
                        if len(logs) >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            # Return empty list if log file doesn't exist yet
            pass
        
        return logs

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """
        Get a list of cases flagged for review.
        
        Returns:
            A list of cases that need review
        """
        return self.get_logs(action='flag_incorrect')

# Example usage
def create_audit_entry(user_id: str, action: str, resource_id: str, **kwargs) -> str:
    """
    Helper function to create an audit log entry.
    
    Args:
        user_id: The ID of the user performing the action
        action: The action being performed
        resource_id: The ID of the resource being acted upon
        **kwargs: Additional metadata to include in the log entry
        
    Returns:
        The request_id of the log entry
    """
    audit_log = AuditLog()
    return audit_log.log_action(
        user_id=user_id,
        action=action,
        resource_id=resource_id,
        model_version=kwargs.get('model_version'),
        metadata=kwargs
    )