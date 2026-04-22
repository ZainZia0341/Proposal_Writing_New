from __future__ import annotations

import json
import logging
from logging.config import dictConfig
from pathlib import Path

from .config import settings

_LOGGING_CONFIGURED = False

_STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}
_IGNORED_EXTRA_FIELDS = {"color_message"}


class ContextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message = self.formatMessage(record)
        extras = self._collect_extras(record)
        if extras:
            rendered_extras = " | ".join(f"{key}={self._serialize_value(value)}" for key, value in extras.items())
            message = f"{message} | {rendered_extras}"

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if message and not message.endswith("\n"):
                message += "\n"
            message += record.exc_text

        if record.stack_info:
            if message and not message.endswith("\n"):
                message += "\n"
            message += self.formatStack(record.stack_info)

        return message

    def _collect_extras(self, record: logging.LogRecord) -> dict[str, object]:
        extras: dict[str, object] = {}
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key.startswith("_") or key in _IGNORED_EXTRA_FIELDS:
                continue
            extras[key] = value
        return dict(sorted(extras.items()))

    def _serialize_value(self, value: object) -> str:
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return repr(value)
        return str(value)


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "()": ContextFormatter,
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": settings.log_level,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "level": settings.log_level,
                    "filename": str(settings.log_file_path),
                    "maxBytes": settings.log_max_bytes,
                    "backupCount": settings.log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": settings.log_level,
            },
        }
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
