import os
import json
import requests
import datetime
import hmac
import hashlib
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from typing import Dict, List, Any, Optional, Set
import google.generativeai as genai
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# App configuration
class Config:
    # NocoDB Configuration
    NOCODB_URL = os.getenv("NOCODB_URL", "https://km.koogle.sk")
    NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
    ROSTER_TABLE_ID = os.getenv("ROSTER_TABLE_ID", "m1848aw7em1uz9g")
    ATTENDANCE_TABLE_ID = os.getenv("ATTENDANCE_TABLE_ID", "mbur916jgs0m7ua")
    UNIDENTIFIED_TABLE_ID = os.getenv("UNIDENTIFIED_TABLE_ID", "mhsf4s0jhp90gnn")

    # Zoom webhook verification
    ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")

    # AI Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    USE_AI_MATCHING = os.getenv("USE_AI_MATCHING", "true").lower() == "true"
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

    # Debugging
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

    # Cache settings
    ROSTER_CACHE_SECONDS = int(os.getenv("ROSTER_CACHE_SECONDS", "600"))

    def __str__(self):
        """Return a string representation of the config (excluding sensitive values)"""
        return {
            "NOCODB_URL": self.NOCODB_URL,
            "ROSTER_TABLE_ID": self.ROSTER_TABLE_ID,
            "ATTENDANCE_TABLE_ID": self.ATTENDANCE_TABLE_ID,
            "UNIDENTIFIED_TABLE_ID": self.UNIDENTIFIED_TABLE_ID,
            "USE_AI_MATCHING": self.USE_AI_MATCHING,
            "CONFIDENCE_THRESHOLD": self.CONFIDENCE_THRESHOLD,
            "DEBUG_MODE": self.DEBUG_MODE,
            "ROSTER_CACHE_SECONDS": self.ROSTER_CACHE_SECONDS
        }.__str__()

# Create config instance
config = Config()

# Configure Gemini if API key is available
if config.GOOGLE_API_KEY:
    genai.configure(api_key=config.GOOGLE_API_KEY)
else:
    print("WARNING: No Google API key provided. AI matching will be disabled.")
    config.USE_AI_MATCHING = False

class AttendanceProcessor:
    def __init__(self):
        self.roster_cache = []
        self.roster_last_updated = None
        self.cache_lifetime = config.ROSTER_CACHE_SECONDS

        # Initialize Gemini model if AI matching is enabled
        if config.USE_AI_MATCHING:
            try:
                self.model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash-001",
                    system_instruction="""
                    You are an attendance matching assistant. Your job is to match participant names from
                    Zoom meetings with their official names in a roster database.
                    - Consider common variations, nicknames, and misspellings.
                    - Consider partial names (first name only, last name only).
                    - Consider spiritual names if available.
                    - Return the best match with confidence score.
                    """
                )
                print("Successfully initialized Gemini AI model")
            except Exception as e:
                print(f"Failed to initialize Gemini AI model: {str(e)}")
                config.USE_AI_MATCHING = False

    async def get_roster(self, force_refresh=False):
        """Fetch the roster from NocoDB, with caching."""
        current_time = datetime.datetime.now()

        # Check if cache is valid
        if (not force_refresh and
            self.roster_last_updated and
            (current_time - self.roster_last_updated).seconds < self.cache_lifetime and
            self.roster_cache):
            return self.roster_cache

        # Fetch from API if cache is invalid
        headers = {"xc-token": config.NOCODB_TOKEN}
        all_roster = []
        page = 1
        limit = 100  # Adjust based on your data size

        # Handle pagination
        while True:
            response = requests.get(
                f"{config.NOCODB_URL}/api/v2/tables/{config.ROSTER_TABLE_ID}/records",
                params={"limit": limit, "offset": (page - 1) * limit},
                headers=headers
            )

            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to get roster: {response.text}")

            data = response.json()
            all_roster.extend(data.get("list", []))

            # Check if we need to fetch more pages
            page_info = data.get("PageInfo", {})
            if page_info.get("isLastPage", True):
                break

            page += 1

        # Update cache
        self.roster_cache = all_roster
        self.roster_last_updated = current_time

        return all_roster

    async def mark_attendance(self, person_id, attendance_date):
        """Mark attendance for a person in the attendance table."""
        headers = {
            "xc-token": config.NOCODB_TOKEN,
            "Content-Type": "application/json"
        }

        # Format date column name (YYYY_MM_DD)
        date_column = attendance_date.replace("-", "_")

        payload = {
            "Id": str(person_id),
            f"{date_column}": "Yes"
        }

        response = requests.patch(
            f"{config.NOCODB_URL}/api/v2/tables/{config.ATTENDANCE_TABLE_ID}/records",
            json=payload,
            headers=headers
        )

        if response.status_code not in [200, 201]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to mark attendance: {response.text}"
            )

        return response.json()

    async def log_unidentified_participant(self, name, join_time, date):
        """Log unidentified participants to the unidentified table."""
        headers = {
            "xc-token": config.NOCODB_TOKEN,
            "Content-Type": "application/json"
        }

        # Format time for better readability
        join_time_formatted = datetime.datetime.fromisoformat(join_time.replace('Z', '+00:00')).strftime("%H:%M")

        payload = {
            "Date": date,
            "joinedTime": join_time_formatted,
            "nameJoinedWith": name
        }

        response = requests.post(
            f"{config.NOCODB_URL}/api/v2/tables/{config.UNIDENTIFIED_TABLE_ID}/records",
            json=payload,
            headers=headers
        )

        if response.status_code not in [200, 201]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to log unidentified participant: {response.text}"
            )

        return response.json()

    async def match_participant_with_roster(self, participant_name, roster):
        """
        Use Gemini AI to match participant names with the roster.
        Returns the matched person ID and confidence score.

        This simplified version doesn't use function calling but instead
        relies on structured JSON output in the response.
        """
        # Extract relevant roster information for matching
        roster_info = []
        for person in roster:
            person_info = {
                "Id": person.get("Id"),
                "firstName": person.get("firstName", ""),
                "lastName": person.get("lastName", ""),
                "spiritualName": person.get("spiritualName"),
                "fullName": f"{person.get('firstName', '')} {person.get('lastName', '')}".strip()
            }
            roster_info.append(person_info)

        # Create the prompt for Gemini
        prompt = f"""
        I need to match a Zoom participant with their entry in our roster database.

        Zoom participant name: "{participant_name}"

        Roster database (showing ID, firstName, lastName, and spiritualName if available):
        {json.dumps(roster_info, indent=2)}

        Find the best match for the participant, considering:
        1. The participant might use only their first name or last name
        2. The participant might use a nickname or variation of their name
        3. The participant might use their spiritual name instead
        4. The participant name might have typos or spelling variations
        5. The participant might join with a completely different name (family member's device, etc.)

        If the confidence is below 0.6, report no match found.

        Provide your answer in this JSON format (and only this format, no explanations outside the JSON):
        {{
            "matchedPersonId": ID_number_or_null,
            "confidence": confidence_score_between_0_and_1,
            "reasoning": "brief explanation of your matching decision"
        }}

        Example:
        {{
            "matchedPersonId": 42,
            "confidence": 0.85,
            "reasoning": "The participant name 'John S.' is likely a shortened version of 'John Smith' (ID 42)"
        }}

        Or for no match:
        {{
            "matchedPersonId": null,
            "confidence": 0.3,
            "reasoning": "No convincing match found for 'XYZ User'"
        }}
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json",
                },
            )

            # Get the text response
            text_response = response.text

            # Try to parse JSON from the response
            try:
                # Strip any potential markdown code block syntax
                cleaned_text = text_response.replace("```json", "").replace("```", "").strip()
                match_data = json.loads(cleaned_text)

                # Validate the expected fields
                if isinstance(match_data, dict) and "matchedPersonId" in match_data:
                    return match_data
                else:
                    raise ValueError("Invalid response format")

            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON from response: {e}")
                print(f"Response text: {text_response}")

                # Try to extract JSON with regex as fallback
                import re
                json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                if json_match:
                    try:
                        match_data = json.loads(json_match.group(0))
                        if isinstance(match_data, dict) and "matchedPersonId" in match_data:
                            return match_data
                    except json.JSONDecodeError:
                        pass

                # Return a default response if parsing fails
                return {
                    "matchedPersonId": None,
                    "confidence": 0,
                    "reasoning": f"Failed to parse response: {str(e)}"
                }

        except Exception as e:
            print(f"Error in AI matching: {str(e)}")
            return {
                "matchedPersonId": None,
                "confidence": 0,
                "reasoning": f"Error in AI processing: {str(e)}"
            }

    def simple_name_matching(self, participant_name, roster):
        """
        A simple fallback name matching algorithm that doesn't rely on AI.
        Returns the matched person ID and confidence score.
        """
        participant_name = participant_name.lower()
        best_match = None
        best_score = 0
        reasoning = ""

        for person in roster:
            person_id = person.get("Id")
            first_name = person.get("firstName", "").lower()
            last_name = person.get("lastName", "").lower()
            spiritual_name = person.get("spiritualName", "").lower() if person.get("spiritualName") else ""
            full_name = f"{first_name} {last_name}".strip()

            # Check exact matches first (high confidence)
            if participant_name == full_name:
                return {"matchedPersonId": person_id, "confidence": 0.95, "reasoning": "Exact full name match"}

            if spiritual_name and participant_name == spiritual_name:
                return {"matchedPersonId": person_id, "confidence": 0.9, "reasoning": "Exact spiritual name match"}

            if participant_name == first_name:
                return {"matchedPersonId": person_id, "confidence": 0.85, "reasoning": "Exact first name match"}

            if participant_name == last_name:
                return {"matchedPersonId": person_id, "confidence": 0.8, "reasoning": "Exact last name match"}

            # Check if participant name is contained within any name fields
            if spiritual_name and participant_name in spiritual_name:
                score = 0.75
                if score > best_score:
                    best_score = score
                    best_match = person_id
                    reasoning = "Partial spiritual name match"

            if participant_name in full_name:
                score = 0.7
                if score > best_score:
                    best_score = score
                    best_match = person_id
                    reasoning = "Partial full name match"

            if first_name in participant_name or last_name in participant_name:
                score = 0.65
                if score > best_score:
                    best_score = score
                    best_match = person_id
                    reasoning = "Name contained in participant name"

            # Check if any part of participant name is in any name fields
            name_parts = participant_name.split()
            for part in name_parts:
                if part in first_name or part in last_name or (spiritual_name and part in spiritual_name):
                    score = 0.6
                    if score > best_score:
                        best_score = score
                        best_match = person_id
                        reasoning = f"Partial match on name component: {part}"

        if best_match and best_score >= 0.6:
            return {"matchedPersonId": best_match, "confidence": best_score, "reasoning": reasoning}
        else:
            return {"matchedPersonId": None, "confidence": best_score, "reasoning": "No confident match found"}

    async def process_participant_joined(self, webhook_data):
        """Process participant joined event and handle attendance marking."""
        if "payload" not in webhook_data or "object" not in webhook_data["payload"]:
            return {"status": "error", "message": "Invalid webhook data format"}

        obj = webhook_data["payload"]["object"]
        participant = obj.get("participant", {})

        participant_name = participant.get("user_name", "Unknown")
        join_time = participant.get("join_time")

        # Get today's date in YYYY-MM-DD format from the join_time
        if join_time:
            today_date = join_time.split("T")[0]  # Extract YYYY-MM-DD from ISO format
        else:
            # Fallback to current date if join_time is not available
            today_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Get roster list
        roster = await self.get_roster()

        # Try Gemini AI first if enabled
        if config.USE_AI_MATCHING:
            try:
                match_result = await self.match_participant_with_roster(participant_name, roster)

                # Check if we got a valid result from Gemini
                if "matchedPersonId" in match_result:
                    person_id = match_result.get("matchedPersonId")
                    confidence = match_result.get("confidence", 0)
                    reasoning = match_result.get("reasoning", "")

                    # Log the AI matching attempt
                    print(f"AI Matching '{participant_name}': ID={person_id}, Confidence={confidence}")
                else:
                    # If Gemini didn't return a valid structure, use fallback
                    raise ValueError("Invalid AI matching result structure")
            except Exception as e:
                # Log the error and fall back to simple matching
                print(f"AI matching failed, using fallback: {str(e)}")
                match_result = self.simple_name_matching(participant_name, roster)
                person_id = match_result.get("matchedPersonId")
                confidence = match_result.get("confidence", 0)
                reasoning = match_result.get("reasoning", "") + " (via fallback matching)"

                # Log the fallback matching attempt
                print(f"Fallback matching '{participant_name}': ID={person_id}, Confidence={confidence}, Reason={reasoning}")
        else:
            # AI matching is disabled, use simple matching directly
            match_result = self.simple_name_matching(participant_name, roster)
            person_id = match_result.get("matchedPersonId")
            confidence = match_result.get("confidence", 0)
            reasoning = match_result.get("reasoning", "")

            # Log the matching attempt
            print(f"Simple matching '{participant_name}': ID={person_id}, Confidence={confidence}, Reason={reasoning}")

        if person_id and confidence >= config.CONFIDENCE_THRESHOLD:
            # Found a match with good confidence - mark attendance
            try:
                attendance_result = await self.mark_attendance(person_id, today_date)
                return {
                    "status": "success",
                    "action": "marked_attendance",
                    "personId": person_id,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to mark attendance: {str(e)}",
                    "personId": person_id,
                    "confidence": confidence
                }
        else:
            # No good match found - log as unidentified
            try:
                unidentified_result = await self.log_unidentified_participant(
                    participant_name, join_time, today_date
                )
                return {
                    "status": "success",
                    "action": "logged_unidentified",
                    "name": participant_name,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to log unidentified participant: {str(e)}"
                }

# Initialize the processor
attendance_processor = AttendanceProcessor()

def verify_zoom_signature(signature: str, timestamp: str, request_body: bytes) -> bool:
    """Verify Zoom webhook signature."""
    if not config.ZOOM_WEBHOOK_SECRET:
        # Skip verification if secret is not set (for testing)
        return True

    if not signature.startswith("v0="):
        return False

    # Create message string with timestamp and request body
    message = f"v0:{timestamp}:{request_body.decode('utf-8')}"

    # Generate HMAC SHA-256 hash
    expected_signature = hmac.new(
        config.ZOOM_WEBHOOK_SECRET.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Compare signatures
    received_signature = signature[3:]  # Remove 'v0='
    return hmac.compare_digest(received_signature, expected_signature)

@app.post("/zoom/webhook")
async def zoom_webhook(request: Request):
    """Process Zoom webhook events."""
    # Get the raw body for signature verification
    body = await request.body()

    # Verify Zoom webhook signature
    signature = request.headers.get("x-zm-signature", "")
    timestamp = request.headers.get("x-zm-request-timestamp", "")

    if not verify_zoom_signature(signature, timestamp, body):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse webhook data
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Process based on event type
    event_type = data.get("event")

    if event_type == "meeting.participant_joined":
        result = await attendance_processor.process_participant_joined(data)
        return result
    else:
        # For other event types, just acknowledge receipt
        return {"status": "success", "message": f"Event {event_type} received but not processed"}

@app.get("/test")
async def test_endpoint():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/debug")
async def debug_info():
    """Get debug information about the current setup."""
    if not config.DEBUG_MODE:
        return {"status": "Debug mode disabled. Enable by setting DEBUG_MODE=true in .env"}

    try:
        # Get roster size
        roster_count = 0
        try:
            roster = await attendance_processor.get_roster(force_refresh=True)
            roster_count = len(roster)
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

        # Test AI model if enabled
        ai_status = "Disabled"
        if config.USE_AI_MATCHING:
            try:
                # Simple test of the AI model
                response = attendance_processor.model.generate_content("Hello, are you working?")
                ai_status = f"OK - Response: {response.text[:50]}..."
            except Exception as e:
                ai_status = f"Error: {str(e)}"

        return {
            "status": "ok",
            "config": {
                "NOCODB_URL": config.NOCODB_URL,
                "ROSTER_TABLE_ID": config.ROSTER_TABLE_ID,
                "ATTENDANCE_TABLE_ID": config.ATTENDANCE_TABLE_ID,
                "UNIDENTIFIED_TABLE_ID": config.UNIDENTIFIED_TABLE_ID,
                "USE_AI_MATCHING": config.USE_AI_MATCHING,
                "CONFIDENCE_THRESHOLD": config.CONFIDENCE_THRESHOLD,
                "DEBUG_MODE": config.DEBUG_MODE,
                "ROSTER_CACHE_SECONDS": config.ROSTER_CACHE_SECONDS
            },
            "status_checks": {
                "nocodb_connection": nocodb_status,
                "roster_count": roster_count,
                "ai_status": ai_status,
                "roster_cache_age": f"{(datetime.datetime.now() - (attendance_processor.roster_last_updated or datetime.datetime.now())).seconds} seconds" if attendance_processor.roster_last_updated else "Not cached yet"
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error generating debug info: {str(e)}"}

@app.get("/roster")
async def get_roster():
    """Endpoint to get the roster (for testing)."""
    roster = await attendance_processor.get_roster(force_refresh=True)
    return {"roster_count": len(roster), "first_few": roster[:5]}

@app.post("/refresh-roster")
async def refresh_roster():
    """Force-refresh the roster cache."""
    roster = await attendance_processor.get_roster(force_refresh=True)
    return {"status": "success", "message": f"Roster refreshed with {len(roster)} entries"}

@app.post("/test-matching")
async def test_matching(request: Request):
    """Test the Gemini AI name matching."""
    data = await request.json()
    name = data.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    roster = await attendance_processor.get_roster()

    results = {}

    # Try AI matching if enabled
    if config.USE_AI_MATCHING:
        try:
            ai_match = await attendance_processor.match_participant_with_roster(name, roster)
            results["ai_match"] = ai_match
        except Exception as e:
            results["ai_match_error"] = str(e)

    # Always include simple matching for comparison
    simple_match = attendance_processor.simple_name_matching(name, roster)
    results["simple_match"] = simple_match

    # Include person details if we have a match from either method
    person_id = None
    if config.USE_AI_MATCHING and "ai_match" in results and results["ai_match"].get("matchedPersonId"):
        person_id = results["ai_match"]["matchedPersonId"]
    elif results["simple_match"].get("matchedPersonId"):
        person_id = results["simple_match"]["matchedPersonId"]

    if person_id:
        person_details = next((p for p in roster if p.get("Id") == person_id), None)
        results["person_details"] = person_details

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188)