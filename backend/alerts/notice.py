"""One-off email notices (tournament status changes, student promotions).
No-op — only logged — when SMTP isn't configured."""
from __future__ import annotations

import logging

from backend.alerts.regime_alert import send_email, smtp_config_from_env

logger = logging.getLogger(__name__)


def email_notice(subject: str, body: str) -> bool:
    cfg = smtp_config_from_env()
    if cfg is None:
        logger.info(f"notice (no SMTP configured): {subject}\n{body}")
        return False
    send_email(cfg, subject, body)
    return True
