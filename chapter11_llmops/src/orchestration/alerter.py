from datetime import datetime, timezone
from typing import Any, Dict
from loguru import logger
from pydantic import BaseModel


class AlertNotification(BaseModel):
    timestamp: str
    channel: str
    status: str
    message: str
    details: Dict[str, Any]


class ZenMLAlerter:

    @staticmethod
    def notify_pipeline_status(status: str, pipeline_name: str, details: Dict[str, Any]) -> AlertNotification:
        alert = AlertNotification(
            timestamp=datetime.now(timezone.utc).isoformat(),
            channel="slack",
            status=status,
            message=f"Pipeline '{pipeline_name}' execution concluded with status: {status.upper()}",
            details=details,
        )

        if status == "succeeded":
            logger.success(f"[ALERT SUCCESS] Channel: #{alert.channel} | {alert.message}")
        else:
            logger.error(f"[ALERT FAILURE] Channel: #{alert.channel} | {alert.message} | Details: {details}")

        return alert
