import os
import json
import requests
import datetime
import hmac
import hashlib
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from typing import Dict, List, Any, Optional, Set
import google.generativeai as genai
import asyncio

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# NocoDB Configuration
NOCODB_URL = os.getenv("NOCODB_URL", "https://km.koogle.sk")
NOCODB_TOKEN = os.getenv("NOCODB_TOKEN")
ROSTER_TABLE_ID = "m1848aw7em1uz9g"
ATTENDANCE_TABLE_ID = "mbur916jgs0m7ua"
UNIDENTIFIED_TABLE_ID = "mhsf4s0jhp90gnn"

# Zoom webhook verification
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")

class AttendanceProcessor:
    def __init__(self):
        self.roster_cache = []
        self.roster_last_updated = None
        # Cache lifetime in seconds (10 minutes)
        self.cache_lifetime = 600

        # Initialize Gemini model
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
        headers = {"xc-token": NOCODB_TOKEN}
        all_roster = []
        page = 1
        limit = 100  # Adjust based on your data size

        # Handle pagination
        while True:
            response = requests.get(
                f"{NOCODB_URL}/api/v2/tables/{ROSTER_TABLE_ID}/records",
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
            "xc-token": NOCODB_TOKEN,
            "Content-Type": "application/json"
        }

        # Format date column name (YYYY_MM_DD)
        date_column = attendance_date.replace("-", "_")

        payload = {
            "Id": str(person_id),
            f"{date_column}": "Yes"
        }

        response = requests.patch(
            f"{NOCODB_URL}/api/v2/tables/{ATTENDANCE_TABLE_ID}/records",
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
            "xc-token": NOCODB_TOKEN,
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
            f"{NOCODB_URL}/api/v2/tables/{UNIDENTIFIED_TABLE_ID}/records",
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

        # Prepare function definition for structured output
        match_schema = {
            "type": "object",
            "properties": {
                "matchedPersonId": {
                    "type": "integer",
                    "description": "The ID of the matched person from the roster, or null if no match found"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score of the match from 0 to 1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the matching decision"
                }
            },
            "required": ["matchedPersonId", "confidence", "reasoning"]
        }

        # Define function for finding match
        find_match_function = {
            "name": "find_roster_match",
            "description": "Find the best match for a participant name in the roster",
            "parameters": match_schema
        }

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

        If the confidence is below 0.6, report no match found (null ID).
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
                tools=[find_match_function]
            )

            # Extract the function call result
            if hasattr(response, 'candidates') and response.candidates:
                function_calls = response.candidates[0].content.parts[0].function_call
                if function_calls:
                    match_result = function_calls.args
                    return match_result

            # Fallback if structured output isn't available
            return {
                "matchedPersonId": None,
                "confidence": 0,
                "reasoning": "Failed to get structured output from AI model."
            }

        except Exception as e:
            print(f"Error in AI matching: {str(e)}")
            return {
                "matchedPersonId": None,
                "confidence": 0,
                "reasoning": f"Error in AI processing: {str(e)}"
            }

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

        # Use Gemini to match participant with roster
        match_result = await self.match_participant_with_roster(participant_name, roster)

        person_id = match_result.get("matchedPersonId")
        confidence = match_result.get("confidence", 0)
        reasoning = match_result.get("reasoning", "")

        # Log the matching attempt
        print(f"Matching '{participant_name}': ID={person_id}, Confidence={confidence}, Reason={reasoning}")

        if person_id and confidence >= 0.6:
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
    if not ZOOM_WEBHOOK_SECRET:
        # Skip verification if secret is not set (for testing)
        return True

    if not signature.startswith("v0="):
        return False

    # Create message string with timestamp and request body
    message = f"v0:{timestamp}:{request_body.decode('utf-8')}"

    # Generate HMAC SHA-256 hash
    expected_signature = hmac.new(
        ZOOM_WEBHOOK_SECRET.encode('utf-8'),
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
    match_result = await attendance_processor.match_participant_with_roster(name, roster)

    # If we have a match, include the person's details
    if match_result.get("matchedPersonId"):
        person_id = match_result["matchedPersonId"]
        person_details = next((p for p in roster if p.get("Id") == person_id), None)
        match_result["personDetails"] = person_details

    return match_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188)