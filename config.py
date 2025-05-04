# File: config.py
# Updated to use webhook_manager for token management

import os
import hmac
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
from webhook_manager import token_manager

# Load environment variables
load_dotenv()

class Config:
    # NocoDB Configuration
    NOCODB_URL = os.getenv("NOCODB_URL", "https://km.koogle.sk")
    NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
    ROSTER_TABLE_ID = os.getenv("ROSTER_TABLE_ID", "m1848aw7em1uz9g")
    ATTENDANCE_TABLE_ID = os.getenv("ATTENDANCE_TABLE_ID", "mbur916jgs0m7ua")
    UNIDENTIFIED_TABLE_ID = os.getenv("UNIDENTIFIED_TABLE_ID", "mhsf4s0jhp90gnn")

    # Webhook Authentication
    WH_AUTH_KEY = os.getenv("WH_AUTH_KEY")
    WH_AUTH_ENABLED = os.getenv("WH_AUTH_ENABLED", "true").lower() == "true"
    WH_AUTH_HEADER_NAME = os.getenv("WH_AUTH_HEADER_NAME", "x-api-key")

    # Zoom custom Headers
    ZOOM_CUSTOM_HEADER_KEY = os.getenv("ZOOM_CUSTOM_HEADER_KEY", "x-zoom-custom-auth")
    ZOOM_CUSTOM_HEADER_VALUE = os.getenv("ZOOM_CUSTOM_HEADER_VALUE")
    ZOOM_CUSTOM_HEADER_ENABLED = os.getenv("ZOOM_CUSTOM_HEADER_ENABLED", "false").lower() == "true"

    # AI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    USE_AI_MATCHING = os.getenv("USE_AI_MATCHING", "true").lower() == "true"
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

    # Vector DB Configuration
    USE_VECTOR_MATCHING = os.getenv("USE_VECTOR_MATCHING", "true").lower() == "true"
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vectordb")

    # Debugging
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Cache settings
    ROSTER_CACHE_SECONDS = int(os.getenv("ROSTER_CACHE_SECONDS", "600"))

    def __init__(self):
        self.load_webhook_secrets_from_env()
        self.validate_api_key_config()
        self.validate_zoom_custom_header_config()
        self.openai_client = None
        if self.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)
        else:
            print("WARNING: No OpenAI API key provided. OpenAI matching will be disabled.")
            self.USE_AI_MATCHING = False
        
        # Still enable vector matching if Jina API key is available
        if not self.JINA_API_KEY and not self.OPENAI_API_KEY:
            print("WARNING: Neither Jina AI nor OpenAI API keys provided. Vector matching will be disabled.")
            self.USE_VECTOR_MATCHING = False

    def validate_api_key_config(self):
        """Validate webhook authentication configuration"""
        if self.WH_AUTH_ENABLED and not self.WH_AUTH_KEY:
            print("WARNING: Webhook authentication is enabled but no auth key is set. Set WH_AUTH_KEY in your .env file.")
            self.WH_AUTH_ENABLED = False

    def verify_zoom_custom_header(self, request_headers):
        """Verify Zoom custom header authentication"""
        if not self.ZOOM_CUSTOM_HEADER_ENABLED:
            return True  # If not enabled, skip this check

        custom_header_value = request_headers.get(self.ZOOM_CUSTOM_HEADER_KEY)
        if not custom_header_value:
            return False

        return custom_header_value == self.ZOOM_CUSTOM_HEADER_VALUE

    def validate_zoom_custom_header_config(self):
        """Validate Zoom custom header configuration"""
        if self.ZOOM_CUSTOM_HEADER_ENABLED and not self.ZOOM_CUSTOM_HEADER_VALUE:
            print("WARNING: Zoom custom header authentication is enabled but no value is set. Set ZOOM_CUSTOM_HEADER_VALUE in your .env file.")
            self.ZOOM_CUSTOM_HEADER_ENABLED = False

    def load_webhook_secrets_from_env(self):
        """Load webhook secrets from environment variables using token_manager."""
        # Use the token manager instead of internal structures
        token_manager.load_from_env()
        
        # For backward compatibility, initialize this property
        self.WEBHOOK_CATEGORIES = set(token_manager.get_categories())
        
        # Log summary of what we found
        for category in self.WEBHOOK_CATEGORIES:
            tokens = token_manager.tokens.get(category, {})
            verified_count = sum(1 for t in tokens.values() if t.get('verified', False))
            print(f"Category '{category}': {len(tokens)} token(s), {verified_count} verified")
        
        print(f"Found {len(self.WEBHOOK_CATEGORIES)} webhook categories: {', '.join(self.WEBHOOK_CATEGORIES)}")

    def verify_webhook_signature(self, category, signature, timestamp, request_body):
        """
        Verify webhook signature using tokens for the specified category.
        Now uses token_manager.
        """
        return token_manager.verify_signature(category, signature, timestamp, request_body)

    def verify_webhook_signature_for_number(self, category, number, signature, timestamp, request_body):
        """
        Verify webhook signature using the token associated with a specific category and number.
        Now uses token_manager.
        """
        return token_manager.verify_signature(category, signature, timestamp, request_body, token_id=number)

    def get_token_by_category_and_number(self, category, number):
        """
        Get the token for a specific category and number.
        Returns (token, is_verified) or (None, False) if not found.
        """
        return token_manager.get_token(category, token_id=number)

    def get_next_unverified_token(self, category=None):
        """
        Get the next unverified token in sequence.
        """
        category = category.lower() if category else None
        
        if not category:
            # Try to find any unverified token
            for cat in token_manager.tokens:
                for token_id, token_info in token_manager.tokens[cat].items():
                    if not token_info.get('verified', False):
                        token, verified = token_manager.get_token(cat, token_id)
                        return token
            
            # If all verified, return first token from any category
            for cat in token_manager.tokens:
                if token_manager.tokens[cat]:
                    first_id = sorted(token_manager.tokens[cat].keys())[0]
                    token, _ = token_manager.get_token(cat, first_id)
                    return token
        else:
            # Look in specific category
            for token_id, token_info in token_manager.tokens.get(category, {}).items():
                if not token_info.get('verified', False):
                    token, _ = token_manager.get_token(category, token_id)
                    return token
                    
            # If all verified in this category, return first token
            if category in token_manager.tokens and token_manager.tokens[category]:
                first_id = sorted(token_manager.tokens[category].keys())[0]
                token, _ = token_manager.get_token(category, first_id)
                return token
                
        return None

    def mark_token_as_verified(self, token):
        """
        Mark a specific token as verified.
        Note: This approach is for backward compatibility only.
        It's better to use token_manager.mark_verified directly.
        """
        # Find the token in the manager
        for category, tokens in token_manager.tokens.items():
            for token_id, token_info in tokens.items():
                if token_info.get('token') == token:
                    token_manager.mark_verified(category, token_id)
                    return True
        return False

    def save_verification_status(self):
        """
        Save verification status to .env file.
        Now handled by token_manager.
        """
        token_manager._update_env_file()

    def __str__(self):
        """Return a string representation of the config (excluding sensitive values)"""
        return {
            "NOCODB_URL": self.NOCODB_URL,
            "ROSTER_TABLE_ID": self.ROSTER_TABLE_ID,
            "ATTENDANCE_TABLE_ID": self.ATTENDANCE_TABLE_ID,
            "UNIDENTIFIED_TABLE_ID": self.UNIDENTIFIED_TABLE_ID,
            "USE_AI_MATCHING": self.USE_AI_MATCHING,
            "USE_VECTOR_MATCHING": self.USE_VECTOR_MATCHING,
            "CONFIDENCE_THRESHOLD": self.CONFIDENCE_THRESHOLD,
            "DEBUG_MODE": self.DEBUG_MODE,
            "WH_AUTH_ENABLED": self.WH_AUTH_ENABLED,
            "ROSTER_CACHE_SECONDS": self.ROSTER_CACHE_SECONDS,
            "WEBHOOK_CATEGORIES": list(self.WEBHOOK_CATEGORIES),
            "TOKENS_COUNT": sum(len(tokens) for tokens in token_manager.tokens.values()),
            "TOKENS_VERIFIED": sum(sum(1 for t in tokens.values() if t.get('verified', False)) for tokens in token_manager.tokens.values()),
            "JINA_ENABLED": bool(self.JINA_API_KEY),
            "OPENAI_ENABLED": bool(self.OPENAI_API_KEY)
        }.__str__()