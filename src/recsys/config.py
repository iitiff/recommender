"""Runtime settings, read from environment variables (or a local .env file).

Secrets (affiliate API keys, session secret) also arrive this way and are
never committed to any repository.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RECSYS_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///data/recsys.db"
    tenant_id: str = "demo"
