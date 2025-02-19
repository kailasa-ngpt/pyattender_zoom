from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import hmac
import hashlib
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

class ZoomAccountManager:
    def __init__(self):
        self.tokens: List[str] = []
        self.verified_tokens: Dict[str, bool] = {}
        self.load_tokens_from_env()

    def load_tokens_from_env(self):
        """Load webhook secret tokens from environment variables"""
        i = 1
        while True:
            token = os.getenv(f'ZOOM_WEBHOOK_SECRET_{i}')
            if token is None:
                break
            self.tokens.append(token)
            self.verified_tokens[token] = False
            i += 1
        
        if not self.tokens:
            raise ValueError("No Zoom webhook secret tokens found in environment variables")

    def get_next_unverified_token(self) -> str:
        """Get the next unverified token in sequence"""
        for token in self.tokens:
            if not self.verified_tokens[token]:
                return token
        return self.tokens[0]  # If all verified, return first token

    def mark_token_as_verified(self, token: str):
        """Mark a specific token as verified"""
        if token in self.verified_tokens:
            self.verified_tokens[token] = True

    def verify_signature(self, signature: str, message: str) -> bool:
        """Try to verify signature with all tokens"""
        if not signature.startswith("v0="):
            return False
        
        received_hash = signature[3:]  # Remove 'v0='
        
        for token in self.tokens:
            expected_hash = generate_hash(message, token)
            if hmac.compare_digest(received_hash, expected_hash):
                return True
        return False

# Initialize the account manager
account_manager = ZoomAccountManager()

def generate_hash(message: str, secret: str) -> str:
    """Generate HMAC SHA-256 hash for a message using the secret token"""
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

@app.post("/zoom/webhook")
async def zoom_webhook(request: Request):
    # Get the raw body content
    body = await request.body()
    body_str = body.decode('utf-8')
    
    try:
        data = json.loads(body_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Case 1: Initial Verification
    if "plainToken" in data:
        plain_token = data["plainToken"]
        current_token = account_manager.get_next_unverified_token()
        encrypted_token = generate_hash(plain_token, current_token)
        
        # Mark this token as verified
        account_manager.mark_token_as_verified(current_token)
        
        # Log verification status (you might want to store this in a database)
        verified_count = sum(account_manager.verified_tokens.values())
        total_count = len(account_manager.tokens)
        print(f"Verified {verified_count} out of {total_count} accounts")
        
        return JSONResponse(content={"encryptedToken": encrypted_token})

    # Case 2: Webhook Event
    zoom_signature = request.headers.get("x-zm-signature")
    zoom_timestamp = request.headers.get("x-zm-request-timestamp")

    if not zoom_signature or not zoom_timestamp:
        raise HTTPException(status_code=400, detail="Missing required headers")

    # Create message string from timestamp + request body
    message = f"{zoom_timestamp}{body_str}"
    
    # Try to verify with any of our tokens
    if not account_manager.verify_signature(zoom_signature, message):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Process the webhook event based on the event type
    event_type = data.get("event")
    
    # Add your webhook handling logic here
    # You might want to handle different events differently
    return {
        "status": "success",
        "message": f"Webhook processed for event: {event_type}"
    }

@app.get("/verification-status")
async def get_verification_status():
    """Endpoint to check verification status of all accounts"""
    verified_count = sum(account_manager.verified_tokens.values())
    total_count = len(account_manager.tokens)
    return {
        "total_accounts": total_count,
        "verified_accounts": verified_count,
        "status_by_token": {
            f"account_{i+1}": verified 
            for i, verified in enumerate(account_manager.verified_tokens.values())
        }
    }

@app.post("/reset-token")
async def reset_token(request: Request):
    """Reset verification status for a specific token"""
    try:
        data = await request.json()
        token = data.get("token")
        
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
            
        # Check if token exists in our list
        if token not in account_manager.tokens:
            raise HTTPException(status_code=404, detail="Token not found")
            
        # Reset the verification status
        account_manager.verified_tokens[token] = False
        
        # Find the account number for this token
        account_number = account_manager.tokens.index(token) + 1
        
        return {
            "status": "success",
            "message": f"Verification status reset for account {account_number}",
            "account_number": account_number
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)