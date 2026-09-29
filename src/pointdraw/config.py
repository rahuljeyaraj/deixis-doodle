"""Runtime settings from environment variables or a local .env file (see .env.example)."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    nebius_api_key: SecretStr = SecretStr("")
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nemotron_model: str = ""

    cosmos_base_url: str = ""
    cosmos_api_key: SecretStr = SecretStr("")
    cosmos_model: str = ""

    speech_base_url: str = ""

    serial_port: str = ""
    camera_index: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()
