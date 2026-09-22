from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Pose Insight"
    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    report_dir: Path = Path("data/reports")
    model_dir: Path = Path("models")
    yolo_model: str = "yolo11n-pose.pt"
    yolo_confidence: float = 0.35
    max_upload_mb: int = 10
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"

    def ensure_directories(self) -> None:
        for directory in (self.upload_dir, self.report_dir, self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()

