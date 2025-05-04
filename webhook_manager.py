# File: webhook_manager.py
# Unified webhook token management with simplified data structure

import os
import re
import hmac
import hashlib
import traceback
from typing import Dict, Tuple, List, Optional, Any

class WebhookTokenManager:
    """
    Unified manager for webhook tokens with simplified data structure and operations.
    """
    
    def __init__(self):
        """Initialize the webhook token manager."""
        # Single structure for all webhook token information
        # Format: {category: {token_id: {'token': value, 'verified': bool}}}
        self.tokens = {}
        
    def load_from_env(self):
        """Load tokens from environment variables."""
        # Reset data
        self.tokens = {}
        
        # Parse environment variables
        pattern = re.compile(r'ZOOM_WEBHOOK_SECRET_([A-Za-z0-9]+)_(\d+)$')
        
        for key, value in os.environ.items():
            match = pattern.match(key)
            if not match:
                continue
                
            category = match.group(1).lower()
            token_id = match.group(2)
            
            # Parse token value and verification status
            if '|' in value:
                token, verified_str = value.split('|', 1)
                verified = verified_str.lower() == 'true'
            else:
                token = value
                verified = False
                
            # Initialize category if needed
            if category not in self.tokens:
                self.tokens[category] = {}
                
            # Store token with all information in one place
            self.tokens[category][token_id] = {
                'token': token,
                'verified': verified
            }
            
        # Log summary (sanitized)
        for category, tokens in self.tokens.items():
            verified_count = sum(1 for t in tokens.values() if t['verified'])
            print(f"Loaded {len(tokens)} tokens for category '{category}', {verified_count} verified")
            
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.tokens.keys())
            
    def get_token(self, category: str, token_id: Optional[str] = None) -> Tuple[Optional[str], bool]:
        """
        Get a token by category and optional ID.
        
        Args:
            category: The webhook category
            token_id: Optional specific token ID
            
        Returns:
            Tuple of (token value, is_verified)
        """
        category = category.lower()
        
        if category not in self.tokens:
            return None, False
            
        if token_id:
            # Get specific token
            if token_id in self.tokens[category]:
                token_info = self.tokens[category][token_id]
                return token_info['token'], token_info['verified']
            return None, False
            
        # Get first token for category (for default behavior)
        if self.tokens[category]:
            first_id = sorted(self.tokens[category].keys())[0]
            token_info = self.tokens[category][first_id]
            return token_info['token'], token_info['verified']
            
        return None, False
        
    def verify_signature(self, category: str, signature: str, timestamp: str, 
                         body: bytes, token_id: Optional[str] = None) -> bool:
        """
        Verify webhook signature.
        
        Args:
            category: The webhook category
            signature: The signature from the webhook header
            timestamp: The timestamp from the webhook header
            body: The raw request body
            token_id: Optional specific token ID
            
        Returns:
            Whether the signature is valid
        """
        # If specific token ID provided, use it directly
        if token_id:
            token, _ = self.get_token(category, token_id)
            if not token:
                return False
                
            # Verify with this specific token
            return self._verify_signature_with_token(signature, timestamp, body, token)
        
        # Otherwise, try all tokens for this category
        category = category.lower()
        if category not in self.tokens:
            return False
            
        # Try each token for this category
        for token_info in self.tokens[category].values():
            token = token_info['token']
            if self._verify_signature_with_token(signature, timestamp, body, token):
                return True
                
        return False
        
    def _verify_signature_with_token(self, signature: str, timestamp: str, 
                                    body: bytes, token: str) -> bool:
        """Helper method to verify signature with a specific token."""
        if not signature.startswith("v0="):
            return False
            
        received_hash = signature[3:]
        message = f"v0:{timestamp}:{body.decode('utf-8')}"
        
        expected_hash = hmac.new(
            token.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(received_hash, expected_hash)
        
    def mark_verified(self, category: str, token_id: str) -> bool:
        """
        Mark a token as verified.
        
        Args:
            category: The webhook category
            token_id: The token ID to mark verified
            
        Returns:
            Whether the operation was successful
        """
        category = category.lower()
        
        if (category not in self.tokens or 
            token_id not in self.tokens[category]):
            return False
            
        self.tokens[category][token_id]['verified'] = True
        self._update_env_file()
        return True
            
    def mark_unverified(self, category: str, token_id: str) -> bool:
        """
        Mark a token as unverified.
        
        Args:
            category: The webhook category
            token_id: The token ID to mark unverified
            
        Returns:
            Whether the operation was successful
        """
        category = category.lower()
        
        if (category not in self.tokens or 
            token_id not in self.tokens[category]):
            return False
            
        self.tokens[category][token_id]['verified'] = False
        self._update_env_file()
        return True
            
    def reset_verification(self, category: Optional[str] = None) -> Dict[str, int]:
        """
        Reset verification status for all tokens in a category or all categories.
        
        Args:
            category: Optional category to reset, or all if None
            
        Returns:
            Dictionary with reset counts by category
        """
        reset_counts = {}
        
        if category:
            # Reset specific category
            category = category.lower()
            if category in self.tokens:
                count = 0
                for token_id in self.tokens[category]:
                    if self.tokens[category][token_id]['verified']:
                        self.tokens[category][token_id]['verified'] = False
                        count += 1
                reset_counts[category] = count
        else:
            # Reset all categories
            for category in self.tokens:
                count = 0
                for token_id in self.tokens[category]:
                    if self.tokens[category][token_id]['verified']:
                        self.tokens[category][token_id]['verified'] = False
                        count += 1
                if count > 0:
                    reset_counts[category] = count
        
        if reset_counts:
            self._update_env_file()
            
        return reset_counts
            
    def _update_env_file(self):
        """Update the .env file with current verification status."""
        try:
            # Read existing .env file
            env_path = os.path.join(os.getcwd(), '.env')
            if not os.path.exists(env_path):
                print("Warning: .env file not found, cannot save verification status")
                return False

            with open(env_path, 'r') as file:
                lines = file.readlines()

            # Update token verification status in .env content
            new_lines = []
            updated_keys = set()

            for line in lines:
                line = line.strip()
                if line.startswith('ZOOM_WEBHOOK_SECRET_'):
                    # Extract key and value
                    key_value = line.split('=', 1)
                    if len(key_value) != 2:
                        new_lines.append(line)
                        continue
                        
                    key, value = key_value
                    
                    # Try to extract category and token_id from key
                    pattern = re.compile(r'ZOOM_WEBHOOK_SECRET_([A-Za-z0-9]+)_(\d+)$')
                    match = pattern.match(key)
                    if not match:
                        new_lines.append(line)
                        continue
                        
                    category = match.group(1).lower()
                    token_id = match.group(2)
                    
                    # Check if we know about this token
                    if (category in self.tokens and 
                        token_id in self.tokens[category]):
                        
                        token_info = self.tokens[category][token_id]
                        token = token_info['token']
                        verified = token_info['verified']
                        
                        # Create updated line
                        new_line = f"{key}={token}|{str(verified).lower()}"
                        new_lines.append(new_line)
                        
                        # Mark as updated
                        updated_keys.add((category, token_id))
                    else:
                        # Keep the line as is
                        new_lines.append(line)
                else:
                    # Non-webhook line, keep as is
                    new_lines.append(line)

            # Write updated content back to .env file
            with open(env_path, 'w') as file:
                file.write('\n'.join(new_lines))

            print(f"Updated verification status in .env file for {len(updated_keys)} tokens")
            return True
            
        except Exception as e:
            print(f"Error updating .env file: {str(e)}")
            traceback.print_exc()
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        Get complete status information for all tokens.
        
        Returns:
            Dictionary with token information by category
        """
        result = {}
        
        for category, tokens in self.tokens.items():
            category_info = {
                "tokens": {},
                "token_count": len(tokens),
                "verified_count": sum(1 for t in tokens.values() if t['verified'])
            }
            
            for token_id, token_info in tokens.items():
                # Include token info with masked token value for security
                token_value = token_info['token']
                masked_token = token_value[:5] + "..." if token_value else None
                
                category_info["tokens"][token_id] = {
                    "token_preview": masked_token,
                    "verified": token_info['verified'],
                    "reset_url": f"/debug/webhooks/reset?category={category}&token_id={token_id}"
                }
                
            result[category] = category_info
            
        return result

# Create global instance
token_manager = WebhookTokenManager()