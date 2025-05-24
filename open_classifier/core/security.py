"""Security utilities for authentication and authorization."""

from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets
import time

from .config import settings
from .exceptions import AuthenticationError

security = HTTPBearer(auto_error=False)

class APIKeyValidator:
    """Validates API keys for authentication."""
    
    def __init__(self):
        # In production, store these in a secure database
        self.valid_keys = {
            "demo-key": {"user_id": "demo", "permissions": ["read", "write"]},
            # Add more keys as needed
        }
    
    def validate_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key and return user info."""
        if api_key in self.valid_keys:
            return self.valid_keys[api_key]
        return None
    
    def generate_key(self, user_id: str, permissions: list) -> str:
        """Generate a new API key."""
        key = secrets.token_urlsafe(32)
        self.valid_keys[key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": time.time()
        }
        return key

api_key_validator = APIKeyValidator()

async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """Verify API key from Authorization header."""
    if not credentials:
        # Allow anonymous access for demo purposes
        return {"user_id": "anonymous", "permissions": ["read"]}
    
    user_info = api_key_validator.validate_key(credentials.credentials)
    if not user_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_info

async def get_current_user(user_info: dict = Depends(verify_api_key)):
    """Get current authenticated user."""
    return user_info

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(user_info: dict = Depends(get_current_user)):
        if permission not in user_info.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user_info
    return permission_checker

def hash_text(text: str) -> str:
    """Create a hash of text for caching purposes."""
    return hashlib.sha256(text.encode()).hexdigest()

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return secrets.token_urlsafe(16) 