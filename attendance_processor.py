# File: attendance_processor.py
# Refactored version with improved error handling, consistent id_number usage and better matching logic

import datetime
import requests
import re
import traceback
from fastapi import HTTPException
from typing import List, Dict, Any, Optional, Tuple, Set
from vector_store import VectorStore
from attendance_queue import AttendanceQueue
from exceptions import AttendanceError

class AttendanceProcessor:
    def __init__(self, config, client=None):
        self.config = config
        self.roster_cache = []
        self.roster_last_updated = None
        self.cache_lifetime = config.ROSTER_CACHE_SECONDS

        # Initialize OpenAI client
        self.client = client
        
        # Initialize vector store
        if config.USE_VECTOR_MATCHING and (config.JINA_API_KEY or client):
            self.vector_store = VectorStore(config)
        else:
            self.vector_store = None
            
        # Initialize attendance queue for failure recovery
        self.attendance_queue = AttendanceQueue()

    async def get_roster(self, force_refresh=False):
        """
        Fetch the roster from NocoDB, with improved caching and error handling.
        Returns list of roster entries (each containing id_number field when available).
        """
        current_time = datetime.datetime.now()

        # Check if cache is valid
        if (not force_refresh and
            self.roster_last_updated and
            (current_time - self.roster_last_updated).seconds < self.cache_lifetime and
            self.roster_cache):
            return self.roster_cache

        # Fetch from API if cache is invalid
        try:
            headers = {"xc-token": self.config.NOCODB_TOKEN}
            all_roster = []
            
            # Set a high limit to get all records in fewer requests
            limit = 1000
            
            # Handle pagination
            page = 1
            
            while True:
                print(f"Fetching roster page {page} with limit {limit}...")
                response = requests.get(
                    f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.ROSTER_TABLE_ID}/records",
                    params={"limit": limit, "offset": (page - 1) * limit},
                    headers=headers
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise AttendanceError(
                        f"Failed to get roster: {error_text}", 
                        status_code=response.status_code
                    )

                data = response.json()
                records_this_page = data.get("list", [])
                
                if not records_this_page:
                    break
                
                all_roster.extend(records_this_page)
                
                print(f"Fetched {len(records_this_page)} records on page {page}")
                print(f"Total records so far: {len(all_roster)}")

                # Check if we need to fetch more pages
                page_info = data.get("PageInfo", {})
                
                # If this is the last page, break
                if page_info.get("isLastPage", True):
                    print("This is the last page")
                    break

                page += 1
                
                # Safety check to avoid infinite loops
                if page > 10:  # Max 10 pages (10,000 records)
                    print("Reached maximum page limit of 10")
                    break

            # Ensure id_number field consistency
            normalized_roster = []
            for entry in all_roster:
                # Always ensure id_number field exists
                if "id_number" not in entry and "Id" in entry:
                    entry["id_number"] = entry["Id"]
                normalized_roster.append(entry)
            
            # Update cache
            self.roster_cache = normalized_roster
            self.roster_last_updated = current_time
            
            print(f"Total roster size: {len(normalized_roster)}")
            return normalized_roster
            
        except Exception as e:
            print(f"Error fetching roster: {str(e)}")
            traceback.print_exc()
            # Return existing cache if available, otherwise re-raise
            if self.roster_cache:
                print("Using cached roster due to fetch error")
                return self.roster_cache
            raise


    async def mark_attendance(self, id_number, attendance_date, matched_person_id):
        """Mark attendance for a person in the attendance table using their id_number and matchedPersonId."""
        if not matched_person_id:
            raise ValueError("Cannot mark attendance: matched_person_id is required")
                
        try:
            headers = {
                "xc-token": self.config.NOCODB_TOKEN,
                "Content-Type": "application/json"
            }

            # Use the date directly as the column name
            date_column = attendance_date

            # First, check if a record with this id_number already exists
            query_response = requests.get(
                f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.ATTENDANCE_TABLE_ID}/records",
                params={"where": f"(id_number,eq,{id_number})"},
                headers=headers
            )
            
            if query_response.status_code == 200:
                records = query_response.json().get("list", [])
                
                if records:
                    # Record exists, update it via PATCH
                    payload = {
                        "Id": str(matched_person_id),  # Always use matched_person_id for Id
                        f"{date_column}": "Yes"
                    }
                    response = requests.patch(
                        f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.ATTENDANCE_TABLE_ID}/records",
                        json=payload,
                        headers=headers
                    )
                else:
                    # Record doesn't exist, create new one
                    payload = {
                        "Id": str(matched_person_id),  # Always use matched_person_id for Id
                        "id_number": str(id_number),  # Added this line
                        f"{date_column}": "Yes"
                    }
                    response = requests.post(
                        f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.ATTENDANCE_TABLE_ID}/records",
                        json=payload,
                        headers=headers
                    )
            else:
                # Query failed, try direct insert

                # For new records (POST)
                payload = {
                    "Id": str(matched_person_id),  # Always use matched_person_id for Id
                    "id_number": str(id_number),  # Add this line
                    f"{date_column}": "Yes"
                }
                response = requests.post(
                    f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.ATTENDANCE_TABLE_ID}/records",
                    json=payload,
                    headers=headers
                )

            if response.status_code not in [200, 201]:
                error_text = response.text
                # Use our custom exception instead of HTTPException
                raise AttendanceError(
                    f"Failed to mark attendance: {error_text}",
                    status_code=response.status_code
                )

            return response.json()

        except Exception as e:
            print(f"Error marking attendance for ID {matched_person_id}: {str(e)}")
            traceback.print_exc()
            raise


    async def log_unidentified_participant(self, name, join_time, date):
        """Log unidentified participants to the unidentified table."""
        try:
            headers = {
                "xc-token": self.config.NOCODB_TOKEN,
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
                f"{self.config.NOCODB_URL}/api/v2/tables/{self.config.UNIDENTIFIED_TABLE_ID}/records",
                json=payload,
                headers=headers
            )

            if response.status_code not in [200, 201]:
                error_text = response.text
                raise AttendanceError(
                    f"Failed to log unidentified participant: {error_text}",
                    status_code=response.status_code
                )

            return response.json()
        except Exception as e:
            print(f"Error logging unidentified participant '{name}': {str(e)}")
            traceback.print_exc()
            raise

    async def match_participant_with_vector(self, participant_name, category='default'):
        """
        Match a participant name against the vector database with enhanced cross-checking.
        Returns a standardized match result format.
        """
        if not self.vector_store:
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": "Vector store not initialized",
                "ambiguous": False,
                "method": "vector"
            }
            
        try:
            # Use vector store's match_name with appropriate threshold
            match_result = await self.vector_store.match_name(
                category, 
                participant_name, 
                threshold=self.config.CONFIDENCE_THRESHOLD,
                min_confidence_gap=0.05  # Require at least 5% confidence gap between top matches
            )
            
            # Return in standardized format
            return {
                "matchedPersonId": match_result.get("matchedPersonId"),
                "id_number": match_result.get("id_number"),
                "confidence": match_result.get("confidence", 0),
                "reasoning": match_result.get("reasoning", ""),
                "ambiguous": match_result.get("ambiguous", False),
                "method": "vector",
                "matches": match_result.get("matches", [])
            }
        except Exception as e:
            print(f"Error in vector matching for '{participant_name}': {str(e)}")
            traceback.print_exc()
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": f"Error in vector matching: {str(e)}",
                "ambiguous": False,
                "method": "vector_error"
            }

    async def match_participant_with_openai(self, participant_name, roster):
        """
        Use OpenAI to match participant names with the roster.
        Returns the matched id_number and confidence score.
        """
        # Skip if OpenAI is not configured
        if not self.client:
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": "OpenAI client not initialized",
                "method": "openai"
            }

        try:
            # Extract relevant roster information for matching
            roster_data = ""
            for person in roster:
                name_parts = []
                if person.get("firstName"):
                    name_parts.append(person.get("firstName"))
                if person.get("lastName"):
                    name_parts.append(person.get("lastName"))
                if person.get("spiritualName"):
                    name_parts.append(f"({person.get('spiritualName')})")
                
                full_name = " ".join(name_parts)
                id_number = person.get("id_number", "")
                roster_data += f"ID: {id_number}, Name: {full_name}\n"

            # Create prompt for OpenAI
            prompt = f"""
I need to match a name from a Zoom meeting attendance to our official roster with extreme precision.
The name from Zoom is: "{participant_name}"

Here is our roster (with ID numbers and names):
{roster_data}

CRITICAL MATCHING RULES - YOU MUST FOLLOW THESE EXACTLY:
1. ZERO WEIGHT: The words "nithya" and "ananda" have ZERO weight in matching - ignore them completely
2. CATEGORY RULE: If "sri" is in a name, it CANNOT be matched to a name with "ma" and vice versa
3. MAIN NAME FOCUS: The main distinctive name (like "chakrarajaswarupini", "tirtharupa") is the PRIMARY matching element
4. EXACT DISTINCTIVE MATCH: The distinctive spiritual name MUST match closely to be considered the same person

Return ONLY the ID number of the best match (with no explanation), or "NO_MATCH". If no reasonable match exists, you MUST return "NO_MATCH".
"""

            # Call OpenAI API with improved system message and settings
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use gpt-4o-mini model
                messages=[
                    {"role": "system", "content": "You are a precise name-matching assistant with expertise in identifying name variations, cultural naming patterns, and determining when a match should or should not be made. You prioritize accuracy over recall and will only provide a match when the evidence is sufficient."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for highly deterministic responses
                max_tokens=100    # Limit response length since we only need the ID
            )

            # Get the text response
            text_response = response.choices[0].message.content.strip()

            # Check if the response is "NO_MATCH"
            if text_response == "NO_MATCH":
                return {
                    "matchedPersonId": None,
                    "id_number": None,
                    "confidence": 0,
                    "reasoning": "No match found by AI",
                    "method": "openai"
                }

            # Try to match the response to a roster entry's id_number
            for person in roster:
                if str(person.get("id_number", "")) == text_response:
                    return {
                        "matchedPersonId": person.get("Id"),  # For backward compatibility
                        "id_number": person.get("id_number"),
                        "confidence": 0.8,  # Default confidence for clear AI matches
                        "reasoning": "AI-based name matching",
                        "method": "openai"
                    }

            # If no direct match, try to extract any alphanumeric token
            id_match = re.search(r'[A-Z0-9]{5,}', text_response)
            if id_match:
                potential_id = id_match.group(0)
                for person in roster:
                    if str(person.get("id_number", "")) == potential_id:
                        return {
                            "matchedPersonId": person.get("Id"),  # For backward compatibility
                            "id_number": person.get("id_number"),
                            "confidence": 0.7,  # Lower confidence for extracted ID
                            "reasoning": "AI-based name matching (extracted ID)",
                            "method": "openai"
                        }

            # If we got here, we couldn't find a clear match
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": f"AI couldn't determine a clear match (response: {text_response})",
                "method": "openai"
            }

        except Exception as e:
            print(f"Error in AI matching for '{participant_name}': {str(e)}")
            traceback.print_exc()
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": f"Error in AI processing: {str(e)}",
                "method": "openai_error"
            }
            
    def simple_name_matching(self, participant_name, roster):
        """
        Simple string-based name matching as a fallback.
        Returns a match result in the standard format.
        """
        try:
            # Normalize participant name for comparison
            participant_name_lower = participant_name.lower()
            participant_parts = set(participant_name_lower.split())
            
            best_match = None
            best_match_score = 0
            
            for person in roster:
                # Extract all name parts
                name_parts = []
                if person.get("firstName"):
                    name_parts.append(person.get("firstName").lower())
                if person.get("lastName"):
                    name_parts.append(person.get("lastName").lower())
                if person.get("spiritualName"):
                    name_parts.extend(person.get("spiritualName").lower().split())
                    
                person_parts = set(name_parts)
                
                # Calculate simple overlap score
                common_parts = participant_parts.intersection(person_parts)
                
                if common_parts:
                    # Score based on ratio of matching parts
                    score = len(common_parts) / max(len(participant_parts), len(person_parts))
                    
                    # Bonus for full name match
                    firstName = person.get('firstName') or ''
                    lastName = person.get('lastName') or ''
                    full_name = f"{firstName.lower()} {lastName.lower()}".strip()
                    
                    if participant_name_lower == full_name:
                        score = 1.0
                        
                    if score > best_match_score:
                        best_match_score = score
                        best_match = person
            
            # Only return matches with minimum threshold
            if best_match_score > 0.4:  # Simple threshold
                return {
                    "matchedPersonId": best_match.get("Id"),  # For backward compatibility
                    "id_number": best_match.get("id_number"),
                    "confidence": best_match_score,
                    "reasoning": f"Simple string matching (score: {best_match_score:.2f})",
                    "method": "simple"
                }
            
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": "No match found with simple string matching",
                "method": "simple"
            }
        except Exception as e:
            print(f"Error in simple matching for '{participant_name}': {str(e)}")
            traceback.print_exc()
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": f"Error in simple matching: {str(e)}",
                "method": "simple_error"
            }
    
    def _extract_person_identifier(self, person):
        """Extract the stable identifier (id_number) from a person record."""
        # Always use id_number for identification
        if "id_number" in person:
            return str(person["id_number"])
        
        # If no id_number is present, but Id is available, use that as fallback
        if "Id" in person:
            print(f"WARNING: Using 'Id' instead of 'id_number' for person identification")
            return str(person["Id"])
        
        # If no id_number, this is an unexpected data format
        print(f"WARNING: Person record missing id_number field: {person.keys()}")
        return None
    
    async def match_participant_with_roster(self, participant_name, roster, category='default'):
        """
        Match a participant name using all available methods in priority order.
        Returns the best match based on confidence score.
        """
        if not participant_name:
            return {
                "matchedPersonId": None,
                "id_number": None,
                "confidence": 0,
                "reasoning": "Empty participant name",
                "method": None
            }
            
        # Define an ordered list of matching methods to try
        results = []
        
        # 1. Try vector matching first (fastest and typically most accurate)
        if self.config.USE_VECTOR_MATCHING and self.vector_store:
            vector_result = await self.match_participant_with_vector(participant_name, category)
            results.append(vector_result)
            
            # Early return if we got a confident match
            if (vector_result.get("id_number") and 
                vector_result.get("confidence", 0) >= self.config.CONFIDENCE_THRESHOLD and
                not vector_result.get("ambiguous", False)):
                return vector_result
        
        # 2. Try AI matching if vector didn't give a confident result
        if self.config.USE_AI_MATCHING and self.client:
            ai_result = await self.match_participant_with_openai(participant_name, roster)
            results.append(ai_result)
            
            # Early return if we got a confident match from AI
            if (ai_result.get("id_number") and 
                ai_result.get("confidence", 0) >= self.config.CONFIDENCE_THRESHOLD):
                return ai_result
        
        # 3. Finally try simple string matching
        simple_result = self.simple_name_matching(participant_name, roster)
        results.append(simple_result)
        
        # Get the highest confidence result that has an id_number
        valid_results = [r for r in results if r.get("id_number") and r.get("confidence", 0) > 0]
        
        if valid_results:
            # Sort by confidence (highest first)
            valid_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            best_match = valid_results[0]
            
            # Check if confidence is above threshold
            if best_match.get("confidence", 0) >= self.config.CONFIDENCE_THRESHOLD:
                return best_match
            else:
                # Return the best match but with an indication it's below threshold
                best_match["reasoning"] = f"{best_match.get('reasoning', '')} (below threshold)"
                best_match["below_threshold"] = True
                return best_match
        
        # No valid match found with any method
        return {
            "matchedPersonId": None,
            "id_number": None,
            "confidence": 0,
            "reasoning": "No match found with any method",
            "method": "none",
            "tried_methods": [r.get("method") for r in results]
        }
    
    async def process_participant_joined(self, data):
        """Process a participant joined event from Zoom with improved error handling."""
        try:
            # Extract relevant data
            payload = data.get("payload", {})
            object_data = payload.get("object", {})
            participant = object_data.get("participant", {})
            
            participant_name = participant.get("user_name", "")
            join_time = participant.get("join_time", "")
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            if not participant_name:
                return {
                    "status": "error",
                    "message": "No participant name found in event data"
                }
            
            # Get the roster
            roster = await self.get_roster()
            
            # Match participant with roster using all available methods
            match_result = await self.match_participant_with_roster(participant_name, roster)
            matched_id_number = match_result.get("id_number")
            
            if matched_id_number:
                try:
                    # Mark attendance using the id_number
                    await self.mark_attendance(matched_id_number, date, match_result.get("matchedPersonId"))
                     
                    # Find the person in the roster to get their name
                    person = next((p for p in roster if str(p.get("id_number", "")) == str(matched_id_number)), {})
                    person_name = f"{person.get('firstName', '')} {person.get('lastName', '')}"
                    
                    return {
                        "status": "success",
                        "message": f"Marked attendance for {person_name}",
                        "joinedAs": participant_name,
                        "matchedTo": person_name,
                        "matchedPersonId": match_result.get("matchedPersonId"),  # For backward compatibility
                        "id_number": matched_id_number,
                        "confidence": match_result.get("confidence"),
                        "reasoning": match_result.get("reasoning"),
                        "method": match_result.get("method"),
                        "ambiguous": match_result.get("ambiguous", False),
                        "below_threshold": match_result.get("below_threshold", False)
                    }
                except Exception as e:
                    print(f"Error marking attendance: {str(e)}")
                    traceback.print_exc()
                    
                    # Add to queue for retry
                    self.attendance_queue.add(matched_id_number, date, participant_name)
                    
                    # Find the person in the roster
                    person = next((p for p in roster if str(p.get("id_number", "")) == str(matched_id_number)), {})
                    person_name = f"{person.get('firstName', '')} {person.get('lastName', '')}" if person else "Unknown Person"
                    
                    return {
                        "status": "queued",
                        "message": f"Found match but attendance marking failed - queued for retry",
                        "joinedAs": participant_name,
                        "matchedTo": person_name,
                        "matchedPersonId": match_result.get("matchedPersonId"),
                        "id_number": matched_id_number,
                        "error": str(e),
                        "queue_stats": self.attendance_queue.get_stats()
                    }
            else:
                try:
                    # Log unidentified participant
                    await self.log_unidentified_participant(participant_name, join_time, date)
                    
                    return {
                        "status": "unidentified",
                        "message": f"Could not identify participant: {participant_name}",
                        "joinedAs": participant_name,
                        "reasoning": match_result.get("reasoning", "No match found"),
                        "tried_methods": match_result.get("tried_methods", [])
                    }
                except Exception as e:
                    print(f"Error logging unidentified participant: {str(e)}")
                    traceback.print_exc()
                    return {
                        "status": "error",
                        "message": f"Could not identify participant and failed to log: {str(e)}",
                        "joinedAs": participant_name,
                        "error": str(e)
                    }
                
        except Exception as e:
            print(f"Error processing participant: {str(e)}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Error processing participant: {str(e)}"
            }