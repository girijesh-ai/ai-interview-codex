"""
Notification Tasks

Background tasks for sending notifications via email, SMS, etc.

Demonstrates:
- Priority queuing
- External service integration
- Retry with exponential backoff
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..app import celery_app

logger = logging.getLogger(__name__)


# ============================================================================
# EMAIL NOTIFICATIONS
# ============================================================================

@celery_app.task(
    bind=True,
    name="tasks.notifications.send_email",
    queue="notifications",
    max_retries=5,
    priority=7  # High priority
)
def send_email(
    self,
    to: str,
    subject: str,
    body: str,
    html: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Send email notification.

    Args:
        self: Task instance
        to: Recipient email
        subject: Email subject
        body: Email body (plain text)
        html: HTML version of body
        attachments: List of attachments

    Returns:
        Send result
    """
    try:
        logger.info(f"Sending email to: {to}")

        # This would integrate with email service (SendGrid, SES, etc.)
        # For now, just simulate

        result = {
            "status": "sent",
            "to": to,
            "subject": subject,
            "sent_at": datetime.now().isoformat(),
            "message_id": f"msg-{self.request.id}"
        }

        logger.info(f"Email sent successfully: {result['message_id']}")

        return result

    except Exception as e:
        logger.error(f"Error sending email: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries * 60)


@celery_app.task(
    name="tasks.notifications.send_bulk_email",
    queue="notifications"
)
def send_bulk_email(
    recipients: List[str],
    subject: str,
    body: str,
    html: Optional[str] = None
) -> Dict[str, Any]:
    """Send bulk email notification.

    Args:
        recipients: List of recipient emails
        subject: Email subject
        body: Email body
        html: HTML version

    Returns:
        Bulk send result
    """
    try:
        logger.info(f"Sending bulk email to {len(recipients)} recipients")

        # Send to each recipient (could use email service bulk API)
        sent_count = 0
        failed = []

        for recipient in recipients:
            try:
                send_email.delay(recipient, subject, body, html)
                sent_count += 1
            except Exception as e:
                failed.append({"email": recipient, "error": str(e)})

        result = {
            "status": "completed",
            "total": len(recipients),
            "sent": sent_count,
            "failed": len(failed),
            "failed_recipients": failed,
            "sent_at": datetime.now().isoformat()
        }

        logger.info(f"Bulk email completed: {sent_count}/{len(recipients)} sent")

        return result

    except Exception as e:
        logger.error(f"Error sending bulk email: {e}")
        raise


# ============================================================================
# CUSTOMER NOTIFICATIONS
# ============================================================================

@celery_app.task(
    name="tasks.notifications.notify_request_received",
    queue="notifications",
    priority=8
)
def notify_request_received(
    customer_email: str,
    request_id: str,
    category: str
) -> Dict[str, Any]:
    """Notify customer that request was received.

    Args:
        customer_email: Customer email
        request_id: Request ID
        category: Request category

    Returns:
        Notification result
    """
    try:
        logger.info(f"Notifying customer about request: {request_id}")

        subject = f"Request Received - {request_id}"
        body = f"""
Dear Customer,

We have received your {category} request (ID: {request_id}).

Our AI agents are processing your request and you should receive a response shortly.

Thank you for your patience!

Best regards,
Support Team
"""

        return send_email.delay(
            to=customer_email,
            subject=subject,
            body=body
        ).get()

    except Exception as e:
        logger.error(f"Error sending request received notification: {e}")
        raise


@celery_app.task(
    name="tasks.notifications.notify_request_completed",
    queue="notifications",
    priority=8
)
def notify_request_completed(
    customer_email: str,
    request_id: str,
    solution: str
) -> Dict[str, Any]:
    """Notify customer that request is completed.

    Args:
        customer_email: Customer email
        request_id: Request ID
        solution: Solution provided

    Returns:
        Notification result
    """
    try:
        logger.info(f"Notifying customer about completion: {request_id}")

        subject = f"Request Completed - {request_id}"
        body = f"""
Dear Customer,

Your request (ID: {request_id}) has been completed!

Solution:
{solution}

If you have any questions, please don't hesitate to reach out.

Best regards,
Support Team
"""

        return send_email.delay(
            to=customer_email,
            subject=subject,
            body=body
        ).get()

    except Exception as e:
        logger.error(f"Error sending completion notification: {e}")
        raise


@celery_app.task(
    name="tasks.notifications.notify_approval_needed",
    queue="notifications",
    priority=9  # Critical
)
def notify_approval_needed(
    approver_email: str,
    request_id: str,
    reason: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Notify approver that approval is needed.

    Args:
        approver_email: Approver email
        request_id: Request ID
        reason: Reason for approval
        context: Request context

    Returns:
        Notification result
    """
    try:
        logger.info(f"Notifying approver about request: {request_id}")

        subject = f"APPROVAL NEEDED - Request {request_id}"
        body = f"""
Hello,

A customer request requires your approval.

Request ID: {request_id}
Reason: {reason}
Priority: {context.get('priority', 'Unknown')}
Category: {context.get('category', 'Unknown')}

Please review and approve/reject at your earliest convenience.

Dashboard: https://dashboard.example.com/approvals/{request_id}

Thank you!
"""

        return send_email.delay(
            to=approver_email,
            subject=subject,
            body=body
        ).get()

    except Exception as e:
        logger.error(f"Error sending approval notification: {e}")
        raise


# ============================================================================
# INTERNAL NOTIFICATIONS
# ============================================================================

@celery_app.task(
    name="tasks.notifications.notify_system_error",
    queue="notifications",
    priority=9  # Critical
)
def notify_system_error(
    error_type: str,
    error_message: str,
    component: str,
    stack_trace: Optional[str] = None
) -> Dict[str, Any]:
    """Notify team about system error.

    Args:
        error_type: Type of error
        error_message: Error message
        component: Component that failed
        stack_trace: Optional stack trace

    Returns:
        Notification result
    """
    try:
        logger.info(f"Notifying about system error in: {component}")

        subject = f"SYSTEM ERROR - {component} - {error_type}"
        body = f"""
SYSTEM ERROR ALERT

Component: {component}
Error Type: {error_type}
Message: {error_message}
Timestamp: {datetime.now().isoformat()}

{f"Stack Trace:\n{stack_trace}" if stack_trace else ""}

Please investigate immediately.
"""

        # Send to ops team
        return send_email.delay(
            to="ops-team@example.com",
            subject=subject,
            body=body
        ).get()

    except Exception as e:
        logger.error(f"Error sending error notification: {e}")
        # Don't retry - avoid infinite loop
        return {"status": "failed", "error": str(e)}


@celery_app.task(
    name="tasks.notifications.send_daily_digest",
    queue="notifications"
)
def send_daily_digest(recipient: str, analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Send daily digest email.

    Args:
        recipient: Recipient email
        analytics: Analytics data

    Returns:
        Send result
    """
    try:
        logger.info(f"Sending daily digest to: {recipient}")

        subject = f"Daily Digest - {analytics.get('date')}"
        body = f"""
Daily System Digest

Date: {analytics.get('date')}

Metrics:
- Total Requests: {analytics.get('metrics', {}).get('total_requests')}
- Completed: {analytics.get('metrics', {}).get('completed_requests')}
- Failed: {analytics.get('metrics', {}).get('failed_requests')}
- Avg Resolution Time: {analytics.get('metrics', {}).get('avg_resolution_time_seconds')}s

Top Categories:
{_format_top_categories(analytics.get('metrics', {}).get('top_categories', {}))}

Have a great day!
"""

        return send_email.delay(
            to=recipient,
            subject=subject,
            body=body
        ).get()

    except Exception as e:
        logger.error(f"Error sending daily digest: {e}")
        raise


def _format_top_categories(categories: Dict[str, int]) -> str:
    """Format top categories for email."""
    lines = []
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {category.title()}: {count}")
    return "\n".join(lines)


# ============================================================================
# SMS NOTIFICATIONS
# ============================================================================

@celery_app.task(
    bind=True,
    name="tasks.notifications.send_sms",
    queue="notifications",
    max_retries=3,
    priority=9
)
def send_sms(
    self,
    phone_number: str,
    message: str
) -> Dict[str, Any]:
    """Send SMS notification.

    Args:
        self: Task instance
        phone_number: Phone number
        message: SMS message

    Returns:
        Send result
    """
    try:
        logger.info(f"Sending SMS to: {phone_number}")

        # This would integrate with SMS service (Twilio, SNS, etc.)
        # For now, just simulate

        result = {
            "status": "sent",
            "phone_number": phone_number,
            "message_length": len(message),
            "sent_at": datetime.now().isoformat(),
            "message_id": f"sms-{self.request.id}"
        }

        logger.info(f"SMS sent successfully: {result['message_id']}")

        return result

    except Exception as e:
        logger.error(f"Error sending SMS: {e}")
        raise self.retry(exc=e, countdown=2 ** self.request.retries * 30)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Send email
    result = send_email.delay(
        to="customer@example.com",
        subject="Test Email",
        body="This is a test email from Celery"
    )
    print(f"Email task: {result.id}")

    # Example: Notify about approval
    result = notify_approval_needed.delay(
        approver_email="manager@example.com",
        request_id="req-123",
        reason="High value refund request",
        context={"priority": 4, "category": "refund"}
    )
    print(f"Approval notification task: {result.id}")

    # Example: Send SMS
    result = send_sms.delay(
        phone_number="+1234567890",
        message="Your request has been completed!"
    )
    print(f"SMS task: {result.id}")
