# File: exceptions.py
# Custom exceptions for better error handling throughout the application

class AttendanceError(Exception):
    """Exception raised for errors in attendance operations."""
    
    def __init__(self, message, status_code=None, details=None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class VectorStoreError(Exception):
    """Exception raised for errors in vector store operations."""
    
    def __init__(self, message, category=None, details=None):
        self.message = message
        self.category = category
        self.details = details
        super().__init__(self.message)


class WebhookError(Exception):
    """Exception raised for errors in webhook processing."""
    
    def __init__(self, message, category=None, status_code=None, details=None):
        self.message = message
        self.category = category
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)