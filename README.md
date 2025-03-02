# Zoom Webhook Application Endpoints

## Health Check
### GET `/test`
```bash
curl http://localhost:8000/test
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
curl -X POST http://localhost:8000/zoom/webhook \
  -H "Content-Type: application/json" \
  -d '{"event": "endpoint.url_validation", "payload": {"plainToken": "your_plain_token"}}'

# Webhook Events
curl -X POST http://localhost:8000/zoom/webhook \
  -H "Content-Type: application/json" \
  -d '{"event": "meeting.started", "payload": {"object": {"uuid": "meeting_uuid", "topic": "Meeting Topic"}}}'
```

## Active Meetings
### GET `/meetings`
```bash
curl http://localhost:8000/meetings
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

## Verification Status
### GET `/verification-status`
```bash
curl http://localhost:8000/verification-status
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
curl -X POST http://localhost:8000/reset-token
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
curl -X POST http://localhost:8000/reset-token \
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

## Generate Reports Manually
### GET `/generate-reports/{meeting_uuid}`
```bash
curl http://localhost:8000/generate-reports/your_meeting_uuid
```
**Response:**
```json
{
    "status": "success",
    "message": "Reports generated for meeting: your_meeting_uuid"
}
```

## File Structure

### Raw Webhooks
- Path: `/Raw/YYYY-MM-DD/meeting_uuid/[timestamp].json`
- Organized by date first, then by meeting UUID
- Each webhook stored with timestamp filename

### Reports
- Path: `/Reports/YYYY-MM-DD/[Topic_ReportType_meeting_uuid].xlsx`
- Organized by date
- Report filenames include topic, report type, and meeting UUID
- Report Types include HourlyReport & EoD

## Environment Configuration

Create a `.env` file with your Zoom webhook secret tokens:
```
ZOOM_WEBHOOK_SECRET_1=your_first_token|false
ZOOM_WEBHOOK_SECRET_2=your_second_token|true
ZOOM_WEBHOOK_SECRET_3=your_third_token|false
```

The `|true` or `|false` indicates whether the token has been verified.