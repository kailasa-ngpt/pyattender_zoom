# File: main.py
# Refactored version with improved error handling, consolidated endpoints, and better background processing

import json
import traceback
import datetime
import hmac
import hashlib
import time
import os
import asyncio
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openai import OpenAI
import requests

from config import Config
from attendance_processor import AttendanceProcessor
from vector_store import VectorStore
from webhook_manager import token_manager
from attendance_queue import AttendanceQueue
from exceptions import AttendanceError, WebhookError, VectorStoreError

# Create config instance
config = Config()

# Initialize FastAPI app
app = FastAPI()

# Configure OpenAI if API key is available
client = None
if config.OPENAI_API_KEY:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    print("OpenAI client initialized for attendance matching")
else:
    print("WARNING: No OpenAI API key provided. OpenAI matching will be disabled.")
    config.USE_AI_MATCHING = False

# Initialize attendance processor
attendance_processor = AttendanceProcessor(config, client)

# Global dictionary to track vectorization status
vectorization_status = {}

# Webhook Authentication Middleware
class WebhookAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth check for category webhook endpoints
        path_parts = request.url.path.split('/')
        if len(path_parts) >= 3 and path_parts[2] == "webhook":
            return await call_next(request)

        # Skip auth check if authentication is disabled
        if not config.WH_AUTH_ENABLED:
            return await call_next(request)

        # Get auth key from request header
        auth_key = request.headers.get(config.WH_AUTH_HEADER_NAME)

        # Validate auth key
        if auth_key != config.WH_AUTH_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication key or missing header"}
            )

        # Continue with the request
        return await call_next(request)

# Add authentication middleware to app
app.add_middleware(WebhookAuthMiddleware)

# Helper function to get vectorization config from request
async def get_vectorization_config(request: Request) -> dict:
    """Extract vectorization config from request body or return default config."""
    try:
        body = await request.body()
        if body:
            return json.loads(body)
    except:
        pass
    
    # Return default config if none provided or error
    return {
        "columns": ["firstName", "lastName", "spiritualName"],
        "combinations": [
            ["fullName", "{0} {1}"],
            ["firstName", "{0}"],
            ["lastName", "{1}"],
            ["fullNameWithSpiritual", "{0} {1} {2}"],
            ["reversedName", "{1}, {0}"]
        ]
    }

# Helper function for consistent time logging
def log_timestamp(message: str):
    """Log a message with timestamp."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

@app.post("/{category}/webhook")
async def category_webhook(category: str, request: Request):
    """Process webhook events for any category with improved token management."""
    try:
        # Get the raw body for signature verification
        body = await request.body()
        body_str = body.decode("utf-8")

        # Log the entire POST request (method, url, headers, body)
        print("================ POST REQUEST ================")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Body: {body_str}")
        print("==============================================")

        # Log the raw request details with more visibility
        log_timestamp(f"WEBHOOK: Raw webhook received for category '{category}': {body_str[:500]}...")
        log_timestamp(f"WEBHOOK: Headers: {dict(request.headers)}")

        # Normalize category to lowercase
        category = category.lower()

        # Custom header verification
        if config.ZOOM_CUSTOM_HEADER_ENABLED:
            custom_header_verified = config.verify_zoom_custom_header(request.headers)
            if not custom_header_verified:
                log_timestamp(f"WEBHOOK: Custom header verification failed")
                raise HTTPException(status_code=401, detail="Invalid custom header authentication")
            else:
                log_timestamp(f"WEBHOOK: Custom header verification successful")

        # Try to parse JSON
        try:
            data = json.loads(body_str)
        except json.JSONDecodeError as e:
            log_timestamp(f"WEBHOOK: JSON parse error: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        event_type = data.get("event")
        log_timestamp(f"WEBHOOK: Event '{event_type}' received for category '{category}'")

        # Extract meeting UUID if available for improved logging
        meeting_uuid = None
        if "payload" in data and "object" in data["payload"]:
            obj = data["payload"]["object"]
            if "uuid" in obj:
                meeting_uuid = obj["uuid"]
                log_timestamp(f"WEBHOOK: Meeting UUID: {meeting_uuid}")

        # Case 1: Handle endpoint validation
        if event_type == "endpoint.url_validation":
            log_timestamp(f"WEBHOOK: Processing endpoint validation for category '{category}'")
            log_timestamp(f"WEBHOOK: Full validation payload: {json.dumps(data)}")

            plain_token = data.get("payload", {}).get("plainToken")
            if not plain_token:
                log_timestamp(f"WEBHOOK: Error: No plain token provided")
                raise HTTPException(status_code=400, detail="No plain token provided")

            # Get tokens for this category
            category_tokens = token_manager.tokens.get(category, {})
            
            if not category_tokens:
                log_timestamp(f"WEBHOOK: Error: No webhook tokens configured for category '{category}'")
                raise HTTPException(status_code=500, detail=f"No webhook tokens configured for category '{category}'")
            
            # Find the token with the lowest ID that is not verified
            # Sort token IDs numerically to ensure we're processing in ascending order
            sorted_token_ids = sorted([int(tid) for tid in category_tokens.keys()])
            
            # First look for an unverified token
            token_id_to_use = None
            for token_id in sorted_token_ids:
                token_id_str = str(token_id)
                if not category_tokens[token_id_str]['verified']:
                    token_id_to_use = token_id_str
                    break
            
            # If no unverified token found, URL validation is redundant
            if token_id_to_use is None:
                log_timestamp(f"WEBHOOK: All tokens for category '{category}' are already verified, no need for validation")
                return {"status": "success", "message": "All tokens already verified for this category"}
            
            # Get the token to use for validation
            token, is_verified = token_manager.get_token(category, token_id_to_use)
            
            log_timestamp(f"WEBHOOK: Using token_id {token_id_to_use}, token (first 5 chars): {token[:5] if token else 'None'}...")
            log_timestamp(f"WEBHOOK: Plain token from webhook: {plain_token}")

            # Generate hash
            def generate_hash(message: str, secret: str) -> str:
                return hmac.new(
                    secret.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()

            encrypted_token = generate_hash(plain_token, token)
            log_timestamp(f"WEBHOOK: Generated encrypted token: {encrypted_token}")

            # Mark this token as verified
            token_manager.mark_verified(category, token_id_to_use)

            # Prepare response
            response = {
                "plainToken": plain_token,
                "encryptedToken": encrypted_token
            }

            log_timestamp(f"WEBHOOK: Validation response: {json.dumps(response)}")
            return JSONResponse(content=response)

        # Case 2: Verify signature for regular webhook events
        signature = request.headers.get("x-zm-signature", "")
        timestamp = request.headers.get("x-zm-request-timestamp", "")

        log_timestamp(f"WEBHOOK: Headers - Signature: {signature}, Timestamp: {timestamp}")

        # IMPORTANT: Check if verification is enabled in config - skip if disabled
        if not config.WH_AUTH_ENABLED:
            log_timestamp(f"WEBHOOK: Signature verification SKIPPED (auth disabled in config)")
        elif signature and timestamp:
            # Verify signature using tokens in ascending numerical order
            category_tokens = token_manager.tokens.get(category, {})
            
            if category_tokens:
                verified = False
                signature_valid = False
                
                # Try tokens in ascending numerical order
                for token_id in sorted([int(tid) for tid in category_tokens.keys()]):
                    token_id_str = str(token_id)
                    token_info = category_tokens[token_id_str]
                    token = token_info['token']
                    
                    # Try to verify using this token
                    signature_valid = token_manager._verify_signature_with_token(signature, timestamp, body, token)
                    log_timestamp(f"WEBHOOK: Signature verification with token {token_id_str} result: {signature_valid}")
                    
                    if signature_valid:
                        verified = True
                        break
                    
                # If we tried all tokens and none worked
                if not verified:
                    log_timestamp(f"WEBHOOK: All tokens failed signature verification - rejecting webhook")
                    raise HTTPException(status_code=401, detail="Invalid signature")
            else:
                log_timestamp(f"WEBHOOK: No tokens configured for category '{category}' - skipping signature verification")
        elif token_manager.tokens.get(category) and config.WH_AUTH_ENABLED:
            # Only enforce signature check if tokens are configured for this category AND auth is enabled
            log_timestamp(f"WEBHOOK: Missing signature or timestamp headers")
            raise HTTPException(status_code=401, detail="Missing signature headers")
        else:
            log_timestamp(f"WEBHOOK: Skipping signature verification - auth disabled or no tokens configured")

        # Process based on event type with enhanced logging
        if event_type == "meeting.participant_joined":
            log_timestamp(f"WEBHOOK: Processing participant joined event")
            
            # Log participant details for debugging
            if "payload" in data and "object" in data["payload"] and "participant" in data["payload"]["object"]:
                participant = data["payload"]["object"]["participant"]
                log_timestamp(f"WEBHOOK: Participant details: {json.dumps(participant)}")
            
            result = await attendance_processor.process_participant_joined(data)
            log_timestamp(f"WEBHOOK: Participant processing result: {json.dumps(result)}")
            return result
        else:
            # For other event types, log details but don't process
            log_timestamp(f"WEBHOOK: Event '{event_type}' received but not processed (only participant_joined is processed)")
            
            # Log event details for debugging
            if "payload" in data and "object" in data["payload"]:
                log_timestamp(f"WEBHOOK: Event payload preview: {json.dumps(data['payload']['object'])[:200]}...")
                
            return {"status": "success", "message": f"Event {event_type} received and logged (only participant_joined is processed)"}
    except HTTPException:
        # Re-raise HTTP exceptions as they are expected
        raise
    except Exception as e:
        error_message = f"WEBHOOK: Unhandled exception in webhook processing: {str(e)}"
        log_timestamp(error_message)
        traceback.print_exc()
        return {"status": "error", "message": error_message}

@app.post("/{category}/vectorize")
async def vectorize_endpoint(
    category: str, 
    request: Request, 
    background_tasks: BackgroundTasks,
    batch_size: int = Query(20, ge=1, le=100),
    force_refresh: bool = False,
    run_async: bool = False
):
    """
    Unified vectorization endpoint that supports:
    - Full refresh (clear + incremental) via force_refresh parameter
    - Asynchronous processing via run_async parameter
    - Configurable batch size
    """
    try:
        # Normalize category to lowercase
        category = category.lower()
        
        # Get vectorization config from request or use defaults
        config_data = await get_vectorization_config(request)
        
        # Check if vector store is initialized
        if not attendance_processor.vector_store:
            attendance_processor.vector_store = VectorStore(config)
        
        # If force_refresh is True, clear the table first
        if force_refresh:
            attendance_processor.vector_store.clear_table(category)
            log_timestamp(f"Cleared vector data for category '{category}' for full refresh")
        
        # Execute in background or synchronously based on parameter
        if run_async:
            # Initialize status tracking
            vectorization_status[category] = {
                "status": "in_progress",
                "start_time": time.time(),
                "config": config_data,
                "progress": 0,
                "processed": 0,
                "total": 0,
                "force_refresh": force_refresh
            }
            
            # Start background task
            background_tasks.add_task(
                background_vectorize_task, 
                category=category,
                config_data=config_data,
                batch_size=batch_size,
                force_refresh=force_refresh
            )
            
            return {
                "status": "accepted", 
                "message": f"{'Full' if force_refresh else 'Incremental'} vectorization started for category '{category}' in the background",
                "category": category,
                "force_refresh": force_refresh
            }
        else:
            # Process synchronously
            # Get roster for this category
            roster = await attendance_processor.get_roster()
            
            # Run vectorization based on whether this is a full refresh or incremental
            if force_refresh:
                # Full vectorization - already cleared the table above
                stats = await attendance_processor.vector_store.vectorize_roster_incremental(
                    category, roster, config_data, batch_size=batch_size
                )
                message = f"Vectorized {stats.get('newly_vectorized', 0)} people from roster for category '{category}'"
            else:
                # Incremental vectorization
                stats = await attendance_processor.vector_store.vectorize_roster_incremental(
                    category, roster, config_data, batch_size=batch_size
                )
                message = f"Vectorized {stats.get('newly_vectorized', 0)} new people from roster for category '{category}' (already had {stats.get('already_vectorized', 0)})"
            
            return {
                "status": "success", 
                "message": message,
                "total_roster_size": len(roster),
                "category": category,
                "stats": stats,
                "force_refresh": force_refresh
            }
            
    except Exception as e:
        log_timestamp(f"Error in vectorization: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Vectorization error: {str(e)}")

async def background_vectorize_task(category, config_data, batch_size=20, force_refresh=False):
    """
    Background task for vectorization with improved progress tracking and error handling.
    """
    try:
        # Get roster for this category
        roster = await attendance_processor.get_roster()
        
        # Update status with total count
        if category in vectorization_status:
            vectorization_status[category]["total"] = len(roster)
        
        # Check if vector store is initialized
        if not attendance_processor.vector_store:
            attendance_processor.vector_store = VectorStore(config)
        
        # Progress tracking callback function
        def update_progress(processed, total, batch_stats=None):
            if category in vectorization_status:
                vectorization_status[category]["progress"] = (processed / total * 100) if total > 0 else 0
                vectorization_status[category]["processed"] = processed
                # Additional status updates if needed
                if batch_stats and isinstance(batch_stats, dict):
                    if "error" in batch_stats:
                        vectorization_status[category]["last_error"] = batch_stats["error"]
        
        # Run vectorization with progress tracking
        log_timestamp(f"Starting {'full' if force_refresh else 'incremental'} vectorization for '{category}' with {len(roster)} entries")
        
        if force_refresh:
            stats = await attendance_processor.vector_store.vectorize_roster(
                category, roster, config_data, batch_size=batch_size,
                progress_callback=update_progress
            )
        else:
            stats = await attendance_processor.vector_store.vectorize_roster_incremental(
                category, roster, config_data, batch_size=batch_size,
                progress_callback=update_progress
            )
        
        # Update status to completed
        vectorization_status[category] = {
            "status": "completed",
            "end_time": time.time(),
            "start_time": vectorization_status.get(category, {}).get("start_time", time.time()),
            "stats": stats,
            "progress": 100,
            "processed": stats.get("processed_count", 0),
            "total": len(roster),
            "force_refresh": force_refresh
        }
        
        count_message = f"{stats.get('vectorized_count', 0)} people" if force_refresh else f"{stats.get('newly_vectorized', 0)} new people (already had {stats.get('already_vectorized', 0)})"
        log_timestamp(f"Background vectorization complete for '{category}': {count_message}")
        
    except Exception as e:
        # Update status to failed
        if category in vectorization_status:
            vectorization_status[category]["status"] = "failed"
            vectorization_status[category]["error"] = str(e)
            vectorization_status[category]["end_time"] = time.time()
        
        log_timestamp(f"Error in background vectorization for '{category}': {str(e)}")
        traceback.print_exc()

@app.get("/{category}/vectorization-status")
async def get_vectorization_status(category: str):
    """Get the status of a vectorization job."""
    category = category.lower()
    
    if category not in vectorization_status:
        return {
            "status": "not_found",
            "message": f"No vectorization job found for category '{category}'"
        }
    
    return vectorization_status[category]

@app.post("/{category}/clear-vectors")
async def clear_vectors_endpoint(category: str):
    """Clear vector data for a category."""
    try:
        # Normalize category to lowercase
        category = category.lower()
        
        # Check if vector store is initialized
        if not attendance_processor.vector_store:
            attendance_processor.vector_store = VectorStore(config)
        
        # Clear the table
        result = attendance_processor.vector_store.clear_table(category)
        
        if result:
            return {"status": "success", "message": f"Cleared vector data for category '{category}'"}
        else:
            return {"status": "warning", "message": f"No vector data found for category '{category}'"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing vectors: {str(e)}")

@app.post("/test-matching")
async def test_matching(request: Request):
    """Test all name matching methods (vector, OpenAI, and simple)."""
    data = await request.json()
    name = data.get("name")
    category = data.get("category", "default")
    
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    roster = await attendance_processor.get_roster()
    results = {}

    # Try vector matching if enabled
    if config.USE_VECTOR_MATCHING and attendance_processor.vector_store:
        try:
            vector_match = await attendance_processor.match_participant_with_vector(name, category)
            results["vector_match"] = vector_match
        except Exception as e:
            results["vector_match_error"] = str(e)
    
    # Try AI matching if enabled
    if config.USE_AI_MATCHING:
        try:
            ai_match = await attendance_processor.match_participant_with_openai(name, roster)
            results["ai_match"] = ai_match
        except Exception as e:
            results["ai_match_error"] = str(e)

    # Always include simple matching for comparison
    simple_match = attendance_processor.simple_name_matching(name, roster)
    if simple_match:
        # Use a default confidence score for simple matching
        results["simple_match"] = {
            "matchedPersonId": simple_match.get("Id"),  # For backward compatibility
            "id_number": simple_match.get("id_number"),
            "confidence": 0.6,  # Default confidence for string matching
            "reasoning": "Simple string matching"
        }
    else:
        results["simple_match"] = {
            "matchedPersonId": None,
            "id_number": None,
            "confidence": 0,
            "reasoning": "No match found"
        }
    
    # Also try the combined approach
    combined_match = await attendance_processor.match_participant_with_roster(name, roster, category)
    results["combined_match"] = combined_match

    # Include person details if we have a match from any method
    id_number = None
    
    # Check combined match first (our preferred approach)
    if combined_match and combined_match.get("id_number"):
        id_number = combined_match.get("id_number")
    
    # If no id_number found yet, check other methods
    if not id_number:
        if config.USE_VECTOR_MATCHING and "vector_match" in results and results["vector_match"].get("id_number"):
            id_number = results["vector_match"]["id_number"]
        elif config.USE_AI_MATCHING and "ai_match" in results and results["ai_match"].get("id_number"):
            id_number = results["ai_match"]["id_number"]
        elif results["simple_match"].get("id_number"):
            id_number = results["simple_match"]["id_number"]

    if id_number:
        person_details = next((p for p in roster if str(p.get("id_number")) == str(id_number)), None)
        results["person_details"] = person_details

    return results

@app.get("/test")
async def test_endpoint():
    """Simple health check endpoint."""
    return {
        "status": "ok", 
        "categories": token_manager.get_categories(),
        "config": {
            "use_ai_matching": config.USE_AI_MATCHING,
            "use_vector_matching": config.USE_VECTOR_MATCHING,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "jina_api_enabled": bool(config.JINA_API_KEY),
            "openai_api_enabled": bool(config.OPENAI_API_KEY)
        }
    }

# Updated debug_info function to use the new validation approach
@app.get("/debug")
async def debug_info():
    """Get debug information about the current setup."""
    if not config.DEBUG_MODE:
        return {"status": "Debug mode disabled. Enable by setting DEBUG_MODE=true in .env"}

    try:
        # Get roster size
        roster_count = 0
        roster_error = None
        try:
            roster = await attendance_processor.get_roster(force_refresh=True)
            roster_count = len(roster)
            
            # Check first few entries for id_number field
            if roster and len(roster) > 0:
                sample_entry = roster[0]
                has_id_number = "id_number" in sample_entry
                sample_id = sample_entry.get("id_number", sample_entry.get("Id", "N/A"))
                sample_fields = list(sample_entry.keys())
            else:
                has_id_number = False
                sample_id = "N/A"
                sample_fields = []
                
        except Exception as e:
            roster_error = str(e)

        # Test NocoDB connection
        nocodb_status = "Unknown"
        try:
            headers = {"xc-token": config.NOCODB_TOKEN}
            response = requests.get(
                f"{config.NOCODB_URL}/api/v2/tables/{config.ROSTER_TABLE_ID}/records",
                params={"limit": 1},
                headers=headers
            )
            nocodb_status = f"OK - Status {response.status_code}"
        except Exception as e:
            nocodb_status = f"Error: {str(e)}"

        # Test OpenAI if enabled
        openai_status = "Disabled"
        if config.USE_AI_MATCHING and client:
            try:
                # Simple test of the OpenAI model
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, are you working?"}
                    ],
                    max_tokens=10
                )
                openai_status = f"OK - Response: {response.choices[0].message.content[:50]}..."
            except Exception as e:
                openai_status = f"Error: {str(e)}"
                
        # Test Jina AI if enabled
        jina_status = "Disabled"
        if config.JINA_API_KEY:
            try:
                import aiohttp
                
                # Run in a synchronous context for the debug endpoint
                import asyncio
                async def test_jina():
                    async with aiohttp.ClientSession() as session:
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {config.JINA_API_KEY}",
                            "Accept": "application/json"
                        }
                        
                        payload = {
                            "model": "jina-embeddings-v3",
                            "input": ["Test embedding"]
                        }
                        
                        async with session.post(
                            "https://api.jina.ai/v1/embeddings",
                            headers=headers,
                            json=payload
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                return f"Error: {error_text}"
                            
                            result = await response.json()
                            return f"OK - Embedding dimension: {len(result['data'][0]['embedding'])}"
                
                jina_status = asyncio.run(test_jina())
            except Exception as e:
                jina_status = f"Error: {str(e)}"

        # Get webhook token status
        webhook_tokens_status = token_manager.get_status()

        # Check vector database if enabled
        vector_status = "Disabled"
        if config.USE_VECTOR_MATCHING and attendance_processor.vector_store:
            try:
                vector_tables = {}
                for category in token_manager.get_categories():
                    try:
                        table = attendance_processor.vector_store.get_table(category)
                        
                        # Check table using vector search instead of pandas
                        zeros = [0.0] * attendance_processor.vector_store.vector_dim
                        results = table.search(zeros, query_type="vector").limit(10).to_list()
                        count = len(results)
                        
                        # Try to get id_number fields to verify structure
                        id_numbers = []
                        for record in results[:5]:  # Get first 5 for sample
                            if 'id_number' in record:
                                id_numbers.append(record['id_number'])
                        
                        # Get the vector dimensions
                        vector_dim = attendance_processor.vector_store.vector_dim
                        vector_tables[category] = {
                            "count": count, 
                            "dimensions": vector_dim,
                            "sample_id_numbers": id_numbers
                        }
                    except Exception as e:
                        vector_tables[category] = {"error": str(e)}
                
                vector_status = {
                    "status": "OK",
                    "tables": vector_tables
                }
            except Exception as e:
                vector_status = f"Error: {str(e)}"

        # Get attendance queue status
        queue_status = attendance_processor.attendance_queue.get_stats()

        # Database access check (simplified from schema validation)
        db_access_status = await validate_database_schema()

        return {
            "status": "ok",
            "config": {
                "NOCODB_URL": config.NOCODB_URL,
                "ROSTER_TABLE_ID": config.ROSTER_TABLE_ID,
                "ATTENDANCE_TABLE_ID": config.ATTENDANCE_TABLE_ID,
                "UNIDENTIFIED_TABLE_ID": config.UNIDENTIFIED_TABLE_ID,
                "USE_AI_MATCHING": config.USE_AI_MATCHING,
                "USE_VECTOR_MATCHING": config.USE_VECTOR_MATCHING,
                "CONFIDENCE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
                "DEBUG_MODE": config.DEBUG_MODE,
                "WH_AUTH_ENABLED": config.WH_AUTH_ENABLED,
                "ROSTER_CACHE_SECONDS": config.ROSTER_CACHE_SECONDS,
                "JINA_API_KEY": bool(config.JINA_API_KEY),
                "OPENAI_API_KEY": bool(config.OPENAI_API_KEY),
            },
            "webhook_categories": token_manager.get_categories(),
            "status_checks": {
                "nocodb_connection": nocodb_status,
                "roster_count": roster_count,
                "roster_error": roster_error,
                "roster_has_id_number": has_id_number,
                "sample_id": sample_id,
                "sample_fields": sample_fields,
                "openai_status": openai_status,
                "jina_status": jina_status,
                "roster_cache_age": f"{(datetime.datetime.now() - (attendance_processor.roster_last_updated or datetime.datetime.now())).seconds} seconds" if attendance_processor.roster_last_updated else "Not cached yet",
                "webhook_tokens": webhook_tokens_status,
                "vector_database": vector_status,
                "attendance_queue": queue_status,
                "database_access": db_access_status
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating debug info: {str(e)}"}

@app.get("/debug/webhooks/status")
async def webhook_status():
    """Get status of all webhook tokens."""
    if not config.DEBUG_MODE:
        return {"status": "Debug mode disabled. Enable by setting DEBUG_MODE=true in .env"}
        
    return {
        "status": "ok",
        "webhook_tokens": token_manager.get_status()
    }

@app.get("/debug/webhooks/reset")
async def reset_webhook_token(category: str = None, token_id: str = None):
    """Reset verification status of webhook tokens."""
    if not config.DEBUG_MODE:
        return {"status": "Debug mode disabled. Enable by setting DEBUG_MODE=true in .env"}
        
    if category and token_id:
        # Reset specific token
        success = token_manager.mark_unverified(category, token_id)
        message = f"Reset verification for {category} token {token_id}" if success else f"Token {token_id} not found for category {category}"
    elif category:
        # Reset all tokens for category
        reset_counts = token_manager.reset_verification(category)
        count = reset_counts.get(category, 0)
        message = f"Reset {count} tokens for category {category}"
    else:
        # Reset all tokens
        reset_counts = token_manager.reset_verification()
        count = sum(reset_counts.values())
        message = f"Reset {count} tokens across {len(reset_counts)} categories"
        
    return {
        "status": "ok",
        "message": message,
        "webhook_tokens": token_manager.get_status()
    }

# Add attendance queue endpoints
@app.get("/debug/attendance-queue")
async def attendance_queue_status():
    """Get status of the attendance queue."""
    if not config.DEBUG_MODE:
        return {"status": "Debug mode disabled. Enable by setting DEBUG_MODE=true in .env"}
        
    stats = attendance_processor.attendance_queue.get_stats()
    
    # Get details of pending records
    pending = attendance_processor.attendance_queue.get_pending()
    
    # Get details of failed records
    failed = attendance_processor.attendance_queue.get_failed()
    
    return {
        "status": "ok",
        "stats": stats,
        "pending": pending[:10],  # Show first 10 pending
        "failed": failed[:10]     # Show first 10 failed
    }

@app.post("/process-queue")
async def process_queue():
    """Process pending attendance records in the queue - one processing attempt."""
    try:
        pending = attendance_processor.attendance_queue.get_pending()
        
        if not pending:
            return {
                "status": "success",
                "message": "No pending attendance records to process"
            }
            
        print(f"Processing {len(pending)} pending attendance records...")
        
        processed = 0
        failed = 0
        results = []
        
        for i, record in enumerate(pending):
            try:
                # Try to mark attendance
                await attendance_processor.mark_attendance(
                    record["id_number"], 
                    record["date"],
                    record["id_number"]  # Using id_number as matchedPersonId as well
                )
                
                # Mark as processed if successful
                attendance_processor.attendance_queue.mark_processed(i)
                processed += 1
                print(f"Successfully processed queued attendance for {record['id_number']}")
                
                results.append({
                    "id_number": record["id_number"],
                    "status": "success"
                })
            except Exception as e:
                # Mark attempt with error - but leave in queue
                attendance_processor.attendance_queue.mark_attempt(i, str(e))
                failed += 1
                print(f"Failed to process queued attendance: {str(e)}")
                
                results.append({
                    "id_number": record["id_number"],
                    "status": "error",
                    "error": str(e)
                })
        
        # Clean up old completed records
        cleaned = attendance_processor.attendance_queue.clean_completed()
        
        return {
            "status": "success",
            "message": f"Processed {len(pending)} pending attendance records",
            "stats": {
                "processed": processed,
                "failed": failed,
                "cleaned": cleaned
            },
            "results": results
        }
    except Exception as e:
        print(f"Error processing attendance queue: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing attendance queue: {str(e)}")

@app.get("/roster")
async def get_roster():
    """Endpoint to get the roster (for testing)."""
    roster = await attendance_processor.get_roster(force_refresh=True)
    
    # Check for id_number field
    has_id_number = False
    if roster and len(roster) > 0:
        has_id_number = "id_number" in roster[0]
    
    first_few = []
    for person in roster[:5]:
        # Create a copy with explicit id_number field
        person_copy = dict(person)
        if "id_number" not in person_copy:
            person_copy["id_number"] = person_copy.get("Id", "Not available")
        first_few.append(person_copy)
        
    return {
        "roster_count": len(roster), 
        "has_id_number_field": has_id_number,
        "first_few": first_few
    }

@app.post("/refresh-roster")
async def refresh_roster():
    """Force-refresh the roster cache."""
    roster = await attendance_processor.get_roster(force_refresh=True)
    return {"status": "success", "message": f"Roster refreshed with {len(roster)} entries"}

# Database schema validation
# Updated validate_database_schema function that doesn't rely on schema endpoints

async def validate_database_schema():
    """
    Simplified database validation that checks table access without trying to validate schema.
    Only verifies if the tables are accessible rather than checking for specific columns.
    """
    try:
        headers = {"xc-token": config.NOCODB_TOKEN}
        schema_status = {
            "roster_table": False,
            "attendance_table": False,
            "unidentified_table": False,
            "issues": []
        }
        
        # Validate roster table by trying to access a single record
        try:
            response = requests.get(
                f"{config.NOCODB_URL}/api/v2/tables/{config.ROSTER_TABLE_ID}/records",
                params={"limit": 1},
                headers=headers
            )
            
            if response.status_code == 200:
                schema_status["roster_table"] = True
                print(f"✓ Roster table accessible")
                
                # Extract available fields from the response for reference
                try:
                    data = response.json()
                    if "list" in data and len(data["list"]) > 0:
                        sample_record = data["list"][0]
                        fields = list(sample_record.keys())
                        print(f"Available roster fields: {fields}")
                        
                        # Check if critical fields exist
                        if "id_number" not in fields:
                            schema_status["issues"].append("Warning: 'id_number' field not found in roster")
                except Exception as e:
                    print(f"Note: Could not inspect roster fields: {str(e)}")
            else:
                print(f"✗ Could not access roster table: {response.status_code}")
                schema_status["issues"].append(f"Could not access roster table: {response.status_code}")
        except Exception as e:
            print(f"Error accessing roster table: {str(e)}")
            schema_status["issues"].append(f"Error accessing roster table: {str(e)}")
        
        # Validate attendance table by simply checking if it's accessible
        try:
            response = requests.get(
                f"{config.NOCODB_URL}/api/v2/tables/{config.ATTENDANCE_TABLE_ID}/records",
                params={"limit": 1},
                headers=headers
            )
            
            if response.status_code == 200:
                schema_status["attendance_table"] = True
                print(f"✓ Attendance table accessible")
            else:
                print(f"✗ Could not access attendance table: {response.status_code}")
                schema_status["issues"].append(f"Could not access attendance table: {response.status_code}")
        except Exception as e:
            print(f"Error accessing attendance table: {str(e)}")
            schema_status["issues"].append(f"Error accessing attendance table: {str(e)}")
            
        # Validate unidentified table by simply checking if it's accessible
        try:
            response = requests.get(
                f"{config.NOCODB_URL}/api/v2/tables/{config.UNIDENTIFIED_TABLE_ID}/records",
                params={"limit": 1},
                headers=headers
            )
            
            if response.status_code == 200:
                schema_status["unidentified_table"] = True
                print(f"✓ Unidentified table accessible")
            else:
                print(f"✗ Could not access unidentified table: {response.status_code}")
                schema_status["issues"].append(f"Could not access unidentified table: {response.status_code}")
        except Exception as e:
            print(f"Error accessing unidentified table: {str(e)}")
            schema_status["issues"].append(f"Error accessing unidentified table: {str(e)}")
        
        # Set critical error if we couldn't access any tables
        if not any([schema_status["roster_table"], 
                    schema_status["attendance_table"], 
                    schema_status["unidentified_table"]]):
            schema_status["critical_error"] = "Could not access any required tables"
            
        return schema_status
        
    except Exception as e:
        print(f"Error during schema validation: {str(e)}")
        traceback.print_exc()
        return {"critical_error": str(e), "issues": [str(e)]}

# Initialize vector store at startup if enabled
@app.on_event("startup")
async def startup_event():
    """Initialize components at startup and perform incremental vectorization."""
    # Initialize webhook token manager
    token_manager.load_from_env()
    print(f"Initialized webhook token manager with {len(token_manager.get_categories())} categories")
    
    # Validate database access (not schema)
    print("Validating database access...")
    schema_status = await validate_database_schema()
    
    # If validation fails critically, log warnings but continue
    if schema_status.get("critical_error"):
        print(f"WARNING: Critical database access issues detected: {schema_status.get('critical_error')}")
        print(f"Application will attempt to continue but may encounter errors during operation")
    
    # NOTE: No automatic attendance queue processing
    print("NOTE: Attendance queue processing is done via the /process-queue endpoint only.")
    
    # Continue with existing vector store initialization...
    if config.USE_VECTOR_MATCHING and (config.JINA_API_KEY or config.OPENAI_API_KEY):
        # Ensure vector store is initialized
        if not attendance_processor.vector_store:
            log_timestamp("Initializing vector store...")
            attendance_processor.vector_store = VectorStore(config)
            log_timestamp(f"Vector store initialized at {config.VECTOR_DB_PATH}")
        
        # Default vectorization config
        default_config = {
            "columns": ["firstName", "lastName", "spiritualName"],
            "combinations": [
                ["fullName", "{0} {1}"],
                ["firstName", "{0}"],
                ["lastName", "{1}"],
                ["fullNameWithSpiritual", "{0} {1} {2}"],
                ["reversedName", "{1}, {0}"]
            ]
        }
        
        # Check each category and perform incremental vectorization
        for category in token_manager.get_categories():
            try:
                log_timestamp(f"Performing incremental vectorization for category '{category}'...")
                
                # Get current roster
                roster = await attendance_processor.get_roster()
                
                # Safety check for empty roster
                if not roster:
                    log_timestamp(f"Warning: Empty roster for category '{category}', skipping vectorization")
                    continue
                
                # Check if roster has id_number field - this is simply informational
                if len(roster) > 0:
                    if "id_number" not in roster[0]:
                        log_timestamp(f"WARNING: Roster entries do not have 'id_number' field. Available fields: {list(roster[0].keys())}")
                
                # Safety check for batch size - prevent division by zero
                batch_size = max(1, min(20, len(roster)))
                
                # Perform incremental vectorization
                stats = await attendance_processor.vector_store.vectorize_roster_incremental(
                    category, roster, default_config, batch_size=batch_size
                )
                
                if stats.get("newly_vectorized", 0) > 0:
                    log_timestamp(f"Vectorized {stats.get('newly_vectorized')} new entries for '{category}'")
                else:
                    log_timestamp(f"No new entries to vectorize for '{category}', already had {stats.get('already_vectorized', 0)}")
            except Exception as e:
                log_timestamp(f"Error during startup vectorization for '{category}': {str(e)}")
                traceback.print_exc()
    else:
        if not config.JINA_API_KEY and not config.OPENAI_API_KEY:
            log_timestamp("WARNING: Neither Jina AI nor OpenAI API keys provided. Vector matching will be disabled.")
        elif not config.USE_VECTOR_MATCHING:
            log_timestamp("Vector matching disabled in configuration")
        else:
            log_timestamp("Vector matching enabled, but no API keys provided")


# If running as a script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)