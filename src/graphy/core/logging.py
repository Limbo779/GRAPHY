"""Structured terminal and file logging setup for GRAPHY."""

from __future__ import annotations

import logging


class ColorFormatter(logging.Formatter):
    """Lightweight ANSI colored log formatter."""

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        time_str = self.formatTime(record, "%H:%M:%S")
        return f"[{time_str}] {color}{record.levelname:<8}{self.RESET} {record.name}: {record.getMessage()}"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root and graphy logger with ANSI formatting."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    logger = logging.getLogger("graphy")
    logger.setLevel(numeric_level)
    return logger
