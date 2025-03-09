# Zoom Webhook Application Endpoints

## Health Check
### GET `/test`
```bash
curl http://localhost:8188/test
```
**Response:**
```json
{
    "status": "ok"
}
```

## Webhook Handler
### POST `/zoom/webhook`
```bash
# Initial Verification
curl -X POST http://localhost:8188/zoom/webhook \
  -H "Content-Type: application/json" \
  -d '{"event": "endpoint.url_validation", "payload": {"plainToken": "your_plain_token"}}'

# Webhook Events
curl -X POST http://localhost:8188/zoom/webhook \
  -H "Content-Type: application/json" \
  -d '{"event": "meeting.started", "payload": {"object": {"uuid": "meeting_uuid", "topic": "Meeting Topic"}}}'
```

## Active Meetings
### GET `/meetings`
```bash
curl http://localhost:8188/meetings
```
**Response:**
```json
{
    "active_meetings": {
        "meeting_uuid1": {
            "topic": "Meeting Topic",
            "start_time": "2025-03-02T17:06:03Z",
            "participant_count": 5
        }
    },
    "total_count": 1
}
```

## Participant Tracking
### GET `/participant-tracking/{meeting_uuid}`
```bash
curl http://localhost:8188/participant-tracking/your_meeting_uuid
```
**Response:**
```json
{
    "meeting_topic": "Meeting Topic",
    "meeting_start": "2025-03-09 14:30:00",
    "meeting_end": "Still active",
    "is_active": true,
    "total_participants": 3,
    "participants": [
        {
            "name": "John Smith",
            "email": "john.smith@example.com",
            "first_join": "2025-03-09 14:30:15",
            "total_time_minutes": 45.5,
            "sessions": [
                {
                    "join_time": "2025-03-09 14:30:15",
                    "leave_time": "2025-03-09 15:15:45",
                    "duration_minutes": 45.5
                }
            ],
            "chat_messages": 4
        },
        {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "first_join": "2025-03-09 14:32:20",
            "total_time_minutes": 43.7,
            "sessions": [
                {
                    "join_time": "2025-03-09 14:32:20",
                    "leave_time": "Still active",
                    "duration_minutes": 43.7
                }
            ],
            "chat_messages": 2
        }
    ]
}
```

## Verification Status
### GET `/verification-status`
```bash
curl http://localhost:8188/verification-status
```
**Response:**
```json
{
    "total_accounts": 3,
    "verified_accounts": 2,
    "status_by_token": {
        "account_1": true,
        "account_2": true,
        "account_3": false
    }
}
```

## Token Reset
### POST `/reset-token`
```bash
# Reset all tokens
curl -X POST http://localhost:8188/reset-token
```
**Response:**
```json
{
    "status": "success",
    "message": "Verification status reset for all 3 accounts",
    "total_accounts": 3
}
```

```bash
# Reset specific token (optional)
curl -X POST http://localhost:8188/reset-token \
  -H "Content-Type: application/json" \
  -d '{"token": "your_webhook_secret_token"}'
```
**Response:**
```json
{
    "status": "success",
    "message": "Verification status reset for account 2",
    "account_number": 2
}
```

## File Structure

### Raw Webhooks
- Path: `/Raw/YYYY-MM-DD/meeting_uuid/[timestamp].json`
- Organized by date first, then by meeting UUID
- Each webhook stored with timestamp filename

## Environment Configuration

Create a `.env` file with your Zoom webhook secret tokens:
```
ZOOM_WEBHOOK_SECRET_1=your_first_token|false
ZOOM_WEBHOOK_SECRET_2=your_second_token|true
ZOOM_WEBHOOK_SECRET_3=your_third_token|false
```

The `|true` or `|false` indicates whether the token has been verified.