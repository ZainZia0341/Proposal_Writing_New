from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI Proposal Generator API")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    google_model_name: str = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-pro")
    mistral_ocr_model: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    google_fallback_models: tuple[str, ...] = _as_csv(
        os.getenv("GOOGLE_FALLBACK_MODELS"),
        default=("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
    )
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "freelance-portfolio-2")
    use_dynamodb: bool = _as_bool(os.getenv("USE_DYNAMODB"), default=False)
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    proposals_table_name: str = os.getenv("USERS_PROPOSALS_TABLE_NAME", "Users-Proposals")
    tasks_table_name: str = os.getenv("TASKS_TABLE_NAME", "Proposal-Tasks")
    retrieval_top_k: int = _as_int(os.getenv("RETRIEVAL_TOP_K"), default=4)
    retrieval_max_retries: int = _as_int(os.getenv("RETRIEVAL_MAX_RETRIES"), default=3)
    summary_trigger_messages: int = _as_int(os.getenv("SUMMARY_TRIGGER_MESSAGES"), default=10)
    recent_messages_to_keep: int = _as_int(os.getenv("RECENT_MESSAGES_TO_KEEP"), default=10)
    api_timeout_seconds: int = _as_int(os.getenv("API_TIMEOUT_SECONDS"), default=29)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    log_file_name: str = os.getenv("LOG_FILE_NAME", "app.log")
    log_max_bytes: int = _as_int(os.getenv("LOG_MAX_BYTES"), default=1_048_576)
    log_backup_count: int = _as_int(os.getenv("LOG_BACKUP_COUNT"), default=3)

    @property
    def log_file_path(self) -> Path:
        return self.log_dir / self.log_file_name


settings = Settings()
