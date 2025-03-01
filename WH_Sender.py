import customtkinter as ctk
import requests
import json
import subprocess
import random
from datetime import datetime, timezone
from urllib3.exceptions import ConnectTimeoutError, ReadTimeoutError
from requests.exceptions import RequestException
import uuid

class WebhookSender(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Sample data for randomization
        self.sample_names = [
            "John Smith", "Emma Johnson", "Michael Williams", "Olivia Brown", 
            "William Jones", "Sophia Garcia", "James Miller", "Isabella Davis",
            "Alexander Wilson", "Charlotte Moore", "Daniel Taylor", "Amelia Anderson",
            "Matthew Thomas", "Harper White", "Ethan Harris", "Evelyn Martin",
            "Benjamin Thompson", "Abigail Martinez", "Samuel Robinson", "Emily Clark"
        ]

        self.sample_topics = [
            "Paramashivoham Level 1", "Weekly Team Meeting", "Project Kickoff",
            "Design Review", "Strategic Planning", "Customer Onboarding",
            "Staff Training", "Product Demo", "Budget Review", "Marketing Strategy"
        ]

        self.sample_messages = [
            "Namaste everyone!", "Hello team, how is everyone doing?",
            "Just wanted to check if everyone can hear me clearly",
            "Please share your thoughts on this topic",
            "Let's take a five-minute break and resume at quarter past",
            "Can someone share the document we were discussing?",
            "Thanks for joining today's session",
            "I'm sharing my screen now, can everyone see it?",
            "Let's go around the room for quick introductions",
            "Please mute yourselves when not speaking"
        ]

        # State tracking
        self.active_meetings = []  # FIFO queue of started meetings
        self.active_participants = {}  # Dictionary of UUID -> participant data
        
        # Window setup
        self.title("Enhanced Webhook Sender")
        self.geometry("900x700")  # Made window bigger for more input fields

        # Create main frame with scrollable content
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(padx=20, pady=20, fill="both", expand=True)

        # Endpoint input
        self.endpoint_label = ctk.CTkLabel(self.main_container, text="Webhook Endpoint:")
        self.endpoint_label.pack(pady=(0, 5))
        self.endpoint_entry = ctk.CTkEntry(self.main_container, width=400)
        self.endpoint_entry.pack(pady=(0, 10))
        self.endpoint_entry.insert(0, "http://localhost:3000/webhook")

        # Create frames for each webhook type
        self.create_meeting_frame()
        self.create_participant_frame()
        self.create_chat_frame()
        
        # Buttons frame
        self.button_frame = ctk.CTkFrame(self.main_container)
        self.button_frame.pack(pady=10, fill="x")
        
        # First row of action buttons
        self.button_frame1 = ctk.CTkFrame(self.button_frame)
        self.button_frame1.pack(pady=5)
        
        self.meeting_started_button = ctk.CTkButton(
            self.button_frame1,
            text="meeting.started",
            command=lambda: self.send_webhook("meeting.started")
        )
        self.meeting_started_button.pack(side="left", padx=5)

        self.meeting_ended_button = ctk.CTkButton(
            self.button_frame1,
            text="meeting.ended",
            command=lambda: self.send_webhook("meeting.ended")
        )
        self.meeting_ended_button.pack(side="left", padx=5)

        # Second row of action buttons
        self.button_frame2 = ctk.CTkFrame(self.button_frame)
        self.button_frame2.pack(pady=5)

        self.participant_joined_button = ctk.CTkButton(
            self.button_frame2,
            text="participant_joined",
            command=lambda: self.send_webhook("meeting.participant_joined")
        )
        self.participant_joined_button.pack(side="left", padx=5)
        
        self.participant_left_button = ctk.CTkButton(
            self.button_frame2,
            text="participant_left",
            command=lambda: self.send_webhook("meeting.participant_left")
        )
        self.participant_left_button.pack(side="left", padx=5)

        self.chat_message_button = ctk.CTkButton(
            self.button_frame2,
            text="chat_message_sent",
            command=lambda: self.send_webhook("meeting.chat_message_sent")
        )
        self.chat_message_button.pack(side="left", padx=5)
        
        # Utility buttons
        self.button_frame3 = ctk.CTkFrame(self.button_frame)
        self.button_frame3.pack(pady=5)
        
        self.randomize_button = ctk.CTkButton(
            self.button_frame3,
            text="🎲 Randomize All Fields",
            command=self.randomize_all_fields
        )
        self.randomize_button.pack(side="left", padx=5)

        self.curl_button = ctk.CTkButton(
            self.button_frame3,
            text="Test with curl",
            command=self.test_with_curl
        )
        self.curl_button.pack(side="left", padx=5)

        # Response display
        self.response_label = ctk.CTkLabel(self.main_container, text="Response Log:")
        self.response_label.pack(pady=(10, 5))
        self.response_text = ctk.CTkTextbox(self.main_container, width=800, height=150)
        self.response_text.pack(pady=(0, 10), fill="both", expand=True)

    def create_meeting_frame(self):
        # Meeting details frame
        self.meeting_frame = ctk.CTkFrame(self.main_container)
        self.meeting_frame.pack(pady=10, fill="x")
        
        self.meeting_label = ctk.CTkLabel(self.meeting_frame, text="Meeting Details", font=("Arial", 14, "bold"))
        self.meeting_label.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        
        # Meeting ID
        self.meeting_id_label = ctk.CTkLabel(self.meeting_frame, text="Meeting ID:")
        self.meeting_id_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.meeting_id_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.meeting_id_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.meeting_id_entry.insert(0, self.generate_meeting_id())
        
        # Meeting Topic
        self.meeting_topic_label = ctk.CTkLabel(self.meeting_frame, text="Topic:")
        self.meeting_topic_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.meeting_topic_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.meeting_topic_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.meeting_topic_entry.insert(0, "Paramashivoham Level 1")
        
        # Meeting UUID
        self.meeting_uuid_label = ctk.CTkLabel(self.meeting_frame, text="UUID:")
        self.meeting_uuid_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.meeting_uuid_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.meeting_uuid_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.meeting_uuid_entry.insert(0, self.generate_uuid())
        
        # Host ID
        self.host_id_label = ctk.CTkLabel(self.meeting_frame, text="Host ID:")
        self.host_id_label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.host_id_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.host_id_entry.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        self.host_id_entry.insert(0, self.generate_host_id())
        
        # Duration
        self.duration_label = ctk.CTkLabel(self.meeting_frame, text="Duration (min):")
        self.duration_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.duration_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.duration_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.duration_entry.insert(0, "1440")
        
        # Timezone
        self.timezone_label = ctk.CTkLabel(self.meeting_frame, text="Timezone:")
        self.timezone_label.grid(row=3, column=2, padx=10, pady=5, sticky="w")
        self.timezone_entry = ctk.CTkEntry(self.meeting_frame, width=200)
        self.timezone_entry.grid(row=3, column=3, padx=10, pady=5, sticky="w")
        self.timezone_entry.insert(0, "America/New_York")
        
        # Randomize button for meeting fields
        self.randomize_meeting_button = ctk.CTkButton(
            self.meeting_frame,
            text="🎲 Randomize Meeting",
            command=self.randomize_meeting_fields
        )
        self.randomize_meeting_button.grid(row=4, column=0, columnspan=4, padx=10, pady=10)

    def create_participant_frame(self):
        # Participant details frame
        self.participant_frame = ctk.CTkFrame(self.main_container)
        self.participant_frame.pack(pady=10, fill="x")
        
        self.participant_label = ctk.CTkLabel(self.participant_frame, text="Participant Details", font=("Arial", 14, "bold"))
        self.participant_label.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        
        # User ID
        self.user_id_label = ctk.CTkLabel(self.participant_frame, text="User ID:")
        self.user_id_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.user_id_entry = ctk.CTkEntry(self.participant_frame, width=200)
        self.user_id_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.user_id_entry.insert(0, str(int(datetime.now().timestamp()))[-8:])
        
        # User Name
        self.user_name_label = ctk.CTkLabel(self.participant_frame, text="User Name:")
        self.user_name_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.user_name_entry = ctk.CTkEntry(self.participant_frame, width=200)
        self.user_name_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.user_name_entry.insert(0, "Participant_1")
        
        # Participant UUID
        self.participant_uuid_label = ctk.CTkLabel(self.participant_frame, text="Participant UUID:")
        self.participant_uuid_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.participant_uuid_entry = ctk.CTkEntry(self.participant_frame, width=200)
        self.participant_uuid_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        self.participant_uuid_entry.insert(0, f"{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:12].upper()}")
        
        # IP Address
        self.ip_label = ctk.CTkLabel(self.participant_frame, text="IP Address:")
        self.ip_label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.ip_entry = ctk.CTkEntry(self.participant_frame, width=200)
        self.ip_entry.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        self.ip_entry.insert(0, "83.108.57.21")
        
        # Email
        self.email_label = ctk.CTkLabel(self.participant_frame, text="Email:")
        self.email_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.email_entry = ctk.CTkEntry(self.participant_frame, width=200)
        self.email_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.email_entry.insert(0, "")
        
        # Randomize button for participant fields
        self.randomize_participant_button = ctk.CTkButton(
            self.participant_frame,
            text="🎲 Randomize Participant",
            command=self.randomize_participant_fields
        )
        self.randomize_participant_button.grid(row=4, column=0, columnspan=4, padx=10, pady=10)

    def create_chat_frame(self):
        # Chat message details frame
        self.chat_frame = ctk.CTkFrame(self.main_container)
        self.chat_frame.pack(pady=10, fill="x")
        
        self.chat_label = ctk.CTkLabel(self.chat_frame, text="Chat Message Details", font=("Arial", 14, "bold"))
        self.chat_label.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky="w")
        
        # Sender Name
        self.sender_name_label = ctk.CTkLabel(self.chat_frame, text="Sender Name:")
        self.sender_name_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.sender_name_entry = ctk.CTkEntry(self.chat_frame, width=200)
        self.sender_name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.sender_name_entry.insert(0, "John Smith")
        
        # Sender Email
        self.sender_email_label = ctk.CTkLabel(self.chat_frame, text="Sender Email:")
        self.sender_email_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.sender_email_entry = ctk.CTkEntry(self.chat_frame, width=200)
        self.sender_email_entry.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.sender_email_entry.insert(0, "john.smith@example.com")
        
        # Sender Type
        self.sender_type_label = ctk.CTkLabel(self.chat_frame, text="Sender Type:")
        self.sender_type_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.sender_type_var = ctk.StringVar(value="host")
        self.sender_type_combobox = ctk.CTkComboBox(
            self.chat_frame, 
            width=200,
            values=["host", "participant"],
            variable=self.sender_type_var
        )
        self.sender_type_combobox.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Recipient Type
        self.recipient_type_label = ctk.CTkLabel(self.chat_frame, text="Recipient Type:")
        self.recipient_type_label.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.recipient_type_var = ctk.StringVar(value="everyone")
        self.recipient_type_combobox = ctk.CTkComboBox(
            self.chat_frame, 
            width=200,
            values=["everyone", "host", "individual"],
            variable=self.recipient_type_var
        )
        self.recipient_type_combobox.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        
        # Message Content
        self.message_content_label = ctk.CTkLabel(self.chat_frame, text="Message Content:")
        self.message_content_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.message_content_entry = ctk.CTkEntry(self.chat_frame, width=600)
        self.message_content_entry.grid(row=3, column=1, columnspan=3, padx=10, pady=5, sticky="we")
        self.message_content_entry.insert(0, "Namaste everyone!")
        
        # Randomize button for chat fields
        self.randomize_chat_button = ctk.CTkButton(
            self.chat_frame,
            text="🎲 Randomize Chat",
            command=self.randomize_chat_fields
        )
        self.randomize_chat_button.grid(row=4, column=0, columnspan=4, padx=10, pady=10)

    def randomize_meeting_fields(self):
        self.meeting_id_entry.delete(0, "end")
        self.meeting_id_entry.insert(0, self.generate_meeting_id())
        
        self.meeting_uuid_entry.delete(0, "end")
        self.meeting_uuid_entry.insert(0, self.generate_uuid())
        
        self.host_id_entry.delete(0, "end")
        self.host_id_entry.insert(0, self.generate_host_id())
        
        self.meeting_topic_entry.delete(0, "end")
        self.meeting_topic_entry.insert(0, random.choice(self.sample_topics))

    def randomize_participant_fields(self):
        self.user_id_entry.delete(0, "end")
        self.user_id_entry.insert(0, str(int(datetime.now().timestamp()))[-8:])
        
        # Get random name
        random_name = random.choice(self.sample_names)
        self.user_name_entry.delete(0, "end")
        self.user_name_entry.insert(0, random_name)
        
        # Generate new UUID
        self.participant_uuid_entry.delete(0, "end")
        self.participant_uuid_entry.insert(0, f"{uuid.uuid4().hex[:8].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:4].upper()}-{uuid.uuid4().hex[:12].upper()}")
        
        # Random IP (simple version)
        random_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        self.ip_entry.delete(0, "end")
        self.ip_entry.insert(0, random_ip)
        
        # Generate email based on name if it's not empty
        if random_name and " " in random_name:
            first, last = random_name.split(" ", 1)
            email = f"{first.lower()}.{last.lower()}@example.com"
            self.email_entry.delete(0, "end")
            self.email_entry.insert(0, email)

    def randomize_chat_fields(self):
        # Get random name
        random_name = random.choice(self.sample_names)
        self.sender_name_entry.delete(0, "end")
        self.sender_name_entry.insert(0, random_name)
        
        # Generate email based on name
        if " " in random_name:
            first, last = random_name.split(" ", 1)
            email = f"{first.lower()}.{last.lower()}@example.com"
            self.sender_email_entry.delete(0, "end")
            self.sender_email_entry.insert(0, email)
        
        # Random sender type
        self.sender_type_var.set(random.choice(["host", "participant"]))
        
        # Random recipient type
        self.recipient_type_var.set(random.choice(["everyone", "host", "individual"]))
        
        # Random message
        self.message_content_entry.delete(0, "end")
        self.message_content_entry.insert(0, random.choice(self.sample_messages))

    def randomize_all_fields(self):
        self.randomize_meeting_fields()
        self.randomize_participant_fields()
        self.randomize_chat_fields()

    def format_error_message(self, error):
        if isinstance(error, (ConnectTimeoutError, ReadTimeoutError)):
            return "Connection timed out. Please check if:\n- The server is running\n- The URL is correct\n- Your internet connection is working"
        elif "Connection refused" in str(error):
            return "Connection refused. The server is not accepting connections."
        elif "Name or service not known" in str(error):
            return "Invalid hostname. Please check the URL."
        else:
            return f"Error: {str(error)}"

    def test_with_curl(self):
        endpoint = self.endpoint_entry.get().strip()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not endpoint.startswith(('http://', 'https://')):
            self.add_to_log(f"[{current_time}] Curl Error: URL must start with http:// or https://\n" + "-" * 50 + "\n")
            return

        try:
            process = subprocess.Popen(['curl', endpoint], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
            stdout, stderr = process.communicate()
            
            log_message = f"[{current_time}] Curl Test\n"
            if stdout:
                log_message += f"Output:\n{stdout}\n"
            if stderr:
                log_message += f"Errors:\n{stderr}\n"
            
            self.add_to_log(log_message + "-" * 50 + "\n")
            
        except Exception as e:
            self.add_to_log(f"[{current_time}] Curl Error: {str(e)}\n" + "-" * 50 + "\n")

    def format_zoom_datetime(self):
        # Format: "2024-01-08T16:23:13Z"
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def generate_meeting_id(self):
        # Generate a unique meeting ID based on timestamp
        return str(int(datetime.now().timestamp()))[-10:]

    def generate_uuid(self):
        # Generate a UUID-like string with == suffix like in the example
        return f"m{str(int(datetime.now().timestamp()))[-12:]}=="

    def generate_host_id(self):
        # Generate a host ID
        return f"h{str(int(datetime.now().timestamp()))[-16:]}=="

    def get_webhook_payload(self, event_type):
        current_time = self.format_zoom_datetime()
        
        if event_type == "meeting.started":
            # Get meeting data from input fields
            meeting_data = {
                "id": self.meeting_id_entry.get(),
                "uuid": self.meeting_uuid_entry.get(),
                "host_id": self.host_id_entry.get(),
                "type": 8,
                "topic": self.meeting_topic_entry.get(),
                "start_time": current_time,
                "duration": int(self.duration_entry.get()),
                "timezone": self.timezone_entry.get()
            }
            self.active_meetings.append(meeting_data.copy())
            
            # Build payload with meeting data
            return {
                "payload": {
                    "account_id": "ruByAP1BSRawJW2I6qfpvQ",
                    "object": {
                        **meeting_data  # Include all meeting data
                    }
                },
                "event_ts": int(datetime.now().timestamp() * 1000),
                "event": event_type
            }

        # For all other events, verify meeting exists first
        if not self.active_meetings and event_type != "meeting.started":
            self.add_to_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: No active meeting for {event_type}. Please send meeting.started first.\n" + "-" * 50 + "\n")
            return None

        # Get current meeting data for base payload
        current_meeting = self.active_meetings[-1] if self.active_meetings else {
            "id": self.meeting_id_entry.get(),
            "uuid": self.meeting_uuid_entry.get(),
            "host_id": self.host_id_entry.get(),
            "type": 8,
            "topic": self.meeting_topic_entry.get(),
            "start_time": current_time,
            "duration": int(self.duration_entry.get()),
            "timezone": self.timezone_entry.get()
        }
        
        base_payload = {
            "payload": {
                "account_id": "ruByAP1BSRawJW2I6qfpvQ",
                "object": {
                    **current_meeting  # Include all current meeting data
                }
            },
            "event_ts": int(datetime.now().timestamp() * 1000),
            "event": event_type
        }

        if event_type == "meeting.ended":
            if not self.active_meetings:
                self.add_to_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: No active meetings to end\n" + "-" * 50 + "\n")
                return None
            meeting_data = self.active_meetings.pop(0)  # Get oldest meeting
            meeting_data["end_time"] = current_time
            base_payload["payload"]["object"].update(meeting_data)
            
        elif event_type == "meeting.participant_joined":
            # Get participant data from input fields
            participant_uuid = self.participant_uuid_entry.get()
            
            # Create new participant data
            participant_data = {
                "user_id": self.user_id_entry.get(),
                "user_name": self.user_name_entry.get(),
                "join_time": current_time,
                "participant_uuid": participant_uuid,
                "names": [self.user_name_entry.get()],  # Add names array with current name
                "public_ip": self.ip_entry.get(),
                "email": self.email_entry.get(),
                "id": "",
                "participant_user_id": ""
            }
            self.active_participants[participant_uuid] = participant_data
            base_payload["payload"]["object"]["participant"] = participant_data.copy()
            
        elif event_type == "meeting.participant_left":
            if not self.active_participants:
                # Use the current input as a fallback even if no active participants
                participant_uuid = self.participant_uuid_entry.get()
                participant_data = {
                    "user_id": self.user_id_entry.get(),
                    "user_name": self.user_name_entry.get(),
                    "participant_uuid": participant_uuid,
                    "names": [self.user_name_entry.get()],
                    "public_ip": self.ip_entry.get(),
                    "email": self.email_entry.get(),
                    "leave_time": current_time,
                    "leave_reason": "left the meeting. Reason : left the meeting"
                }
            else:
                # Get the oldest participant's UUID (first key in dictionary)
                participant_uuid = next(iter(self.active_participants))
                participant_data = self.active_participants.pop(participant_uuid)
                participant_data["leave_time"] = current_time
                participant_data["leave_reason"] = "left the meeting. Reason : left the meeting"
                if "join_time" in participant_data:
                    del participant_data["join_time"]  # Remove join_time for leave events
            
            base_payload["payload"]["object"]["participant"] = participant_data
            
        elif event_type == "meeting.chat_message_sent":
            base_payload["payload"]["object"]["chat_message"] = {
                "date_time": current_time,
                "sender_session_id": f"s{str(int(datetime.now().timestamp()))[-8:]}",
                "sender_name": self.sender_name_entry.get(),
                "sender_email": self.sender_email_entry.get(),
                "sender_type": self.sender_type_var.get(),
                "recipient_type": self.recipient_type_var.get(),
                "message_id": f"m{str(int(datetime.now().timestamp()))[-8:]}",
                "message_content": self.message_content_entry.get()
            }

        return base_payload

    def send_webhook(self, event_type):
        endpoint = self.endpoint_entry.get().strip()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not endpoint.startswith(('http://', 'https://')):
            self.add_to_log(f"[{current_time}] Error for {event_type}: URL must start with http:// or https://")
            return

        payload = self.get_webhook_payload(event_type)
        if payload is None:  # If there was an error generating the payload
            return

        try:
            response = requests.post(endpoint, json=payload, timeout=(5, 5))
            status = f"[{current_time}] {event_type}\nStatus Code: {response.status_code}\n"
            response_text = f"Response: {response.text}\n"
            self.add_to_log(status + response_text + "-" * 50 + "\n")
            
            # Display the sent payload in the log
            payload_pretty = json.dumps(payload, indent=2)
            self.add_to_log(f"Payload:\n{payload_pretty}\n" + "-" * 50 + "\n")
        except RequestException as e:
            error_msg = self.format_error_message(e)
            self.add_to_log(f"[{current_time}] Error for {event_type}:\n{error_msg}\n" + "-" * 50 + "\n")

    def add_to_log(self, message):
        # Add new message at the end
        self.response_text.insert("end", message)
        
        # Scroll to bottom to show latest message
        self.response_text.see("end")

if __name__ == "__main__":
    app = WebhookSender()
    app.mainloop()