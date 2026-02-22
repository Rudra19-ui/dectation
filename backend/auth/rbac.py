import functools
import jwt
from datetime import datetime, timedelta
from flask import request, jsonify, current_app, g
from typing import Dict, List, Optional, Callable, Any

# Define roles and permissions
ROLES = {
    'patient': ['view_own_reports', 'upload_images'],
    'radiologist': ['view_all_reports', 'review_flagged', 'add_annotations'],
    'admin': ['view_all_reports', 'view_audit_logs', 'manage_users', 'view_statistics'],
    'researcher': ['view_anonymized_data', 'export_statistics']
}

class AuthError(Exception):
    """Exception raised for authentication and authorization errors."""
    def __init__(self, error: str, status_code: int):
        self.error = error
        self.status_code = status_code

def get_token_from_request() -> Optional[str]:
    """Extract JWT token from the request header."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    return auth_header.split('Bearer ')[1]

def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token."""
    try:
        # In a real application, get the secret key from environment variables
        secret_key = current_app.config.get('JWT_SECRET_KEY', 'development_secret_key')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        # Check if token is expired
        if 'exp' in payload and datetime.fromtimestamp(payload['exp']) < datetime.now():
            raise AuthError('Token has expired', 401)
            
        return payload
    except jwt.InvalidTokenError:
        raise AuthError('Invalid token', 401)

def generate_token(user_id: str, role: str, expiration: int = 3600) -> str:
    """Generate a JWT token for a user.
    
    Args:
        user_id: The user's ID
        role: The user's role
        expiration: Token expiration time in seconds (default: 1 hour)
        
    Returns:
        JWT token string
    """
    payload = {
        'sub': user_id,
        'role': role,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(seconds=expiration)
    }
    
    # In a real application, get the secret key from environment variables
    secret_key = current_app.config.get('JWT_SECRET_KEY', 'development_secret_key')
    return jwt.encode(payload, secret_key, algorithm='HS256')

def has_permission(user_role: str, required_permission: str) -> bool:
    """Check if a role has a specific permission."""
    if user_role not in ROLES:
        return False
    return required_permission in ROLES[user_role]

def require_auth(permission: Optional[str] = None) -> Callable:
    """Decorator to require authentication and optional permission.
    
    Args:
        permission: The permission required to access the endpoint
        
    Returns:
        Decorated function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            token = get_token_from_request()
            if not token:
                return jsonify({'error': 'Authorization header is missing'}), 401
            
            try:
                payload = decode_token(token)
                
                # Store user info in Flask's g object for the current request
                g.user_id = payload['sub']
                g.user_role = payload['role']
                
                # Check permission if specified
                if permission and not has_permission(g.user_role, permission):
                    return jsonify({
                        'error': f'Permission denied: {permission} is required'
                    }), 403
                    
                return f(*args, **kwargs)
            except AuthError as e:
                return jsonify({'error': e.error}), e.status_code
                
        return decorated
    return decorator

# Example protected route
'''
@app.route('/api/admin/audit-logs', methods=['GET'])
@require_auth('view_audit_logs')
def get_audit_logs():
    # This endpoint is only accessible to users with 'view_audit_logs' permission
    # (typically admins)
    from models.audit import AuditLog
    audit_log = AuditLog()
    logs = audit_log.get_logs(limit=100)
    return jsonify(logs)
'''

# Example login function (in a real app, this would verify credentials against a database)
def login(username: str, password: str) -> Dict[str, Any]:
    """Authenticate a user and return a JWT token.
    
    Args:
        username: The user's username
        password: The user's password
        
    Returns:
        Dict containing token and user info
    """
    # In a real application, verify credentials against a database
    # This is just a mock implementation
    mock_users = {
        'patient1': {'password': 'password1', 'role': 'patient', 'id': 'p12345'},
        'radiologist1': {'password': 'password2', 'role': 'radiologist', 'id': 'r12345'},
        'admin1': {'password': 'password3', 'role': 'admin', 'id': 'a12345'}
    }
    
    if username not in mock_users or mock_users[username]['password'] != password:
        raise AuthError('Invalid credentials', 401)
    
    user = mock_users[username]
    token = generate_token(user['id'], user['role'])
    
    # Log the login action
    from models.audit import create_audit_entry
    create_audit_entry(
        user_id=user['id'],
        action='login',
        resource_id=None,
        ip_address=request.remote_addr
    )
    
    return {
        'token': token,
        'user_id': user['id'],
        'role': user['role'],
        'expires_in': 3600  # 1 hour
    }