# Zoom Webhook Application Endpoints

## Authentication

The application uses API key authentication for all endpoints except the Zoom webhook endpoint. 
To authenticate, include the `x-api-key` header in your requests:

```
x-api-key: your_api_key_here
```

You can configure the API key in the `.env` file:
```
API_KEY=your_api_key_here
API_KEY_ENABLED=true
API_KEY_HEADER_NAME=x-api-key
```

Note: The `/zoom/webhook` endpoint does not require the API key as it needs to be accessible to Zoom.

## Health Check
### GET `/test`
```bash
curl http://localhost:8000/test -H "x-api-key: your_api_key_here"
```
**Response:**
```json
{
    "status": "ok"
}
```

## Webhook Handler
# Multiple Zoom Webhook Endpoints

This application now supports multiple Zoom webhook endpoints with dedicated token verification for each endpoint.

## Webhook Endpoints

### Default Endpoint
- **URL**: `/zoom/webhook`
- **Verification**: Uses the next unverified token or the first token if all are verified
- **Usage**: Backward compatibility with existing integrations

## Configuration

Configure your `.env` file to match the numbered endpoints:

```
# Token for endpoint /zoom/webhook_1
ZOOM_WEBHOOK_SECRET_1=your_first_token|verification_status

# Token for endpoint /zoom/webhook_2
ZOOM_WEBHOOK_SECRET_2=your_second_token|verification_status

# Token for endpoint /zoom/webhook_3
ZOOM_WEBHOOK_SECRET_3=your_third_token|verification_status
```

Each token includes a verification status (`true` or `false`), which is automatically updated when the endpoint is successfully validated.

## Example Usage

### Endpoint Validation
---
```bash
# Validate endpoint
curl -X POST http://localhost:8188/zoom/webhook
  -H "Content-Type: application/json" \
  -d '{"event": "endpoint.url_validation", "payload": {"plainToken": "token_from_zoom"}}'

```

```

## Zoom Configuration

In the Zoom Developer Portal:

1. For the first account/app:
   - Set Event Notification Endpoint URL to: `https://your-domain.com/zoom/webhook_1`

2. For the second account/app:
   - Set Event Notification Endpoint URL to: `https://your-domain.com/zoom/webhook_2`

And so on for additional accounts or applications.

## Checking Verification Status

```bash
curl http://localhost:8188/verification-status -H "x-api-key: your_api_key_here"
```

This will show the verification status of all configured tokens.

## Security Considerations

- Each endpoint uses its own dedicated token for verification
- The system still supports both signature verification and custom header verification
- All security best practices from the original implementation are maintained
---

## Active Meetings
### GET `/meetings`
```bash
curl http://localhost:8000/meetings -H "x-api-key: your_api_key_here"
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
curl http://localhost:8000/verification-status -H "x-api-key: your_api_key_here"
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
curl -X POST http://localhost:8000/reset-token -H "x-api-key: your_api_key_here"
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
  -H "x-api-key: your_api_key_here" \
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
curl http://localhost:8000/generate-reports/your_meeting_uuid -H "x-api-key: your_api_key_here"
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

Create a `.env` file with your configuration:
```
# API Key Authentication
API_KEY=your_api_key_here
API_KEY_ENABLED=true
API_KEY_HEADER_NAME=x-api-key

# Zoom Webhook Secret Tokens
ZOOM_WEBHOOK_SECRET_1=your_first_token|false
ZOOM_WEBHOOK_SECRET_2=your_second_token|true
ZOOM_WEBHOOK_SECRET_3=your_third_token|false
```

The `|true` or `|false` indicates whether the token has been verified.

---

# Setting Up Zoom Custom Header Authentication

This guide explains how to set up the Custom Header authentication method for Zoom webhooks in your application.

## What is Custom Header Authentication?

Custom Header Authentication is one of Zoom's recommended verification methods for webhooks. With this method, when Zoom sends webhook notifications to your application, it includes a custom HTTP header with a predefined value. Your application verifies that the header is present and contains the expected value to ensure the request is legitimately from Zoom.

## Implementation Steps

### 1. Configure Your Zoom Webhook App

1. Log in to the [Zoom App Marketplace Developer Portal](https://marketplace.zoom.us/develop/create)
2. Select your Webhook app (or create a new one)
3. Go to the "Feature" tab and find the "Event Subscriptions" section
4. Under "Verification Token", select "Custom Header" option
5. Enter a header name (e.g., `x-zoom-custom-auth`) and a secure value
6. Save your changes

![Zoom Custom Header Setup](https://marketplace.zoom.us/docs/images/webhooks-custom-header.png)

### 2. Configure Your Application

1. In your `.env` file, set the following values:

```
ZOOM_CUSTOM_HEADER_ENABLED=true
ZOOM_CUSTOM_HEADER_KEY=x-zoom-custom-auth
ZOOM_CUSTOM_HEADER_VALUE=your_custom_header_value_here
```

Make sure the `ZOOM_CUSTOM_HEADER_KEY` and `ZOOM_CUSTOM_HEADER_VALUE` match exactly what you configured in the Zoom Developer Portal.

### 3. Testing Your Setup

1. Deploy your application
2. In the Zoom Developer Portal, click "Validate" to test the endpoint verification
3. Check your application logs to ensure the custom header verification is working properly

## Troubleshooting

If you're experiencing issues with custom header verification:

1. **Ensure header names match exactly**: The header name is case-sensitive. Ensure the `ZOOM_CUSTOM_HEADER_KEY` in your application matches the header name in Zoom.

2. **Check header value**: The header value must match exactly. Check for any extra spaces or special characters.

3. **Enable debug mode**: Set `DEBUG_MODE=true` in your `.env` file for more detailed logs.

4. **Inspect raw webhook headers**: Use the `/verification-status` endpoint to check your current configuration.

## Security Considerations

- Use a complex, random value for your custom header to prevent guessing
- Keep your custom header value confidential, treating it like any other secret or API key
- Consider using both custom header verification and signature verification for enhanced security

## Additional Information

For more details on Zoom webhook verification methods, see the [official Zoom documentation](https://developers.zoom.us/docs/api/rest/webhook-reference/#verify-webhook-events).