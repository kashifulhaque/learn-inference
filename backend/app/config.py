"""Application settings, read from the environment or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Auth
    app_password: str = "mochimochi"
    session_secret: str = "dev-only-insecure-secret"
    session_days: int = 30
    cookie_secure: bool = False

    # App
    data_dir: Path = REPO_ROOT / "data"
    public_url: str = "http://localhost:8000"
    content_dir: Path = REPO_ROOT / "content"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"

    # GPU providers
    gpu_provider: str = "modal"
    modal_token_id: str = ""
    modal_token_secret: str = ""
    modal_app_name: str = "learn-inference"
    modal_gpu: str = "A100-80GB"
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""

    # Model
    hf_token: str = ""
    model_id: str = "Qwen/Qwen3.8-27B"
    small_model_id: str = "Qwen/Qwen3-0.6B"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
