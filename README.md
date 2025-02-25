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
  -d '{"plainToken": "your_plain_token"}'

# Webhook Events
curl -X POST http://localhost:8000/zoom/webhook \
  -H "Content-Type: application/json" \
  -H "x-zm-signature: v0=hash_value" \
  -H "x-zm-request-timestamp: timestamp" \
  -d '{"event": "event_type", ...}'
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