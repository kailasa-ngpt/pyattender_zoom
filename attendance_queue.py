# File: attendance_queue.py
# Persistent queue for tracking and retrying attendance operations

import os
import json
import datetime
import traceback

class AttendanceQueue:
    """A persistent queue for tracking attendance operations."""
    
    def __init__(self, file_path="./data/attendance_queue.json"):
        self.file_path = file_path
        self.queue = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Load existing queue if available
        self._load_queue()
    
    def _load_queue(self):
        """Load queue from file if it exists."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.queue = json.load(f)
                print(f"Loaded {len(self.queue)} pending attendance records from {self.file_path}")
        except Exception as e:
            print(f"Error loading attendance queue: {str(e)}")
            traceback.print_exc()
            self.queue = []
    
    def _save_queue(self):
        """Save queue to file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.queue, f, indent=2)
        except Exception as e:
            print(f"Error saving attendance queue: {str(e)}")
            traceback.print_exc()
    
    def add(self, id_number, date, participant_name=None):
        """Add an attendance record to the queue."""
        record = {
            "id_number": id_number,
            "date": date,
            "participant_name": participant_name,
            "created_at": datetime.datetime.now().isoformat(),
            "attempts": 0,
            "last_attempt": None,
            "last_error": None,
            "status": "pending"
        }
        
        self.queue.append(record)
        self._save_queue()
        return record
    
    def get_pending(self, max_attempts=5):
        """Get records that need processing."""
        return [r for r in self.queue if r["status"] == "pending" and r["attempts"] < max_attempts]
    
    def get_failed(self):
        """Get permanently failed records (exceeded max attempts)."""
        return [r for r in self.queue if r["status"] == "pending" and r["attempts"] >= 5]
    
    def mark_processed(self, index):
        """Mark a record as successfully processed."""
        if 0 <= index < len(self.queue):
            self.queue[index]["status"] = "completed"
            self.queue[index]["completed_at"] = datetime.datetime.now().isoformat()
            self._save_queue()
    
    def mark_attempt(self, index, error=None):
        """Mark an attempted processing with optional error."""
        if 0 <= index < len(self.queue):
            self.queue[index]["attempts"] += 1
            self.queue[index]["last_attempt"] = datetime.datetime.now().isoformat()
            
            if error:
                self.queue[index]["last_error"] = str(error)
                
            self._save_queue()
    
    def clean_completed(self, older_than_days=7):
        """Remove completed records older than the specified number of days."""
        if not self.queue:
            return 0
            
        now = datetime.datetime.now()
        cleaned = 0
        
        new_queue = []
        for record in self.queue:
            if record["status"] == "completed":
                completed_at = datetime.datetime.fromisoformat(record["completed_at"])
                age_days = (now - completed_at).total_seconds() / (60 * 60 * 24)
                
                if age_days > older_than_days:
                    cleaned += 1
                    continue
                    
            new_queue.append(record)
        
        if cleaned > 0:
            self.queue = new_queue
            self._save_queue()
            
        return cleaned
    
    def get_stats(self):
        """Get queue statistics."""
        if not self.queue:
            return {
                "total": 0,
                "pending": 0,
                "completed": 0,
                "failed": 0
            }
            
        pending = sum(1 for r in self.queue if r["status"] == "pending" and r["attempts"] < 5)
        completed = sum(1 for r in self.queue if r["status"] == "completed")
        failed = sum(1 for r in self.queue if r["status"] == "pending" and r["attempts"] >= 5)
        
        return {
            "total": len(self.queue),
            "pending": pending,
            "completed": completed,
            "failed": failed
        }