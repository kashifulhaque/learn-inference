"""Application settings, read from the environment or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Caddy sets these headers only after Authelia authorizes the request.
    # The app rejects requests without the shared proxy secret.
    sso_proxy_secret: str = ""
    # Maps Authelia usernames to existing progress identities, e.g.
    # "ifkash:enderboi,ananya:michu"; unmapped users keep their username.
    sso_users: str = ""

    @property
    def sso_names(self) -> dict[str, str]:
        pairs = (item.split(":", 1) for item in self.sso_users.split(",") if ":" in item)
        return {
            sso.strip().lower(): name.strip()
            for sso, name in pairs
            if sso.strip() and name.strip()
        }

    # App
    data_dir: Path = REPO_ROOT / "data"
    public_url: str = "http://localhost:8000"
    content_dir: Path = REPO_ROOT / "content"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    # The PDF book and its manifest, written by frontend/scripts/build-book.mjs.
    # The Docker image runs that script in its `book` stage and copies the
    # result here.
    book_dir: Path = REPO_ROOT / "book"

    # GPU providers
    gpu_provider: str = "runpod"
    # The accelerator the labs ask for, for display.
    gpu_type: str = "A100-80GB"
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
