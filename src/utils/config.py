"""Configuration loader for the project."""
import os
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_env: str = "development"
    app_name: str = "investment-transaction-monitor"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./dev.db"
    redis_url: str = "redis://localhost:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "investment-transaction-monitoring"

    # Model settings
    model_registry_path: str = "models/trained"
    anomaly_threshold: float = 0.75
    risk_score_high: float = 0.80
    risk_score_medium: float = 0.50

    # API security
    secret_key: str = "dev-secret-key"
    api_key_header: str = "X-API-Key"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache()
def get_config() -> dict:
    return load_yaml_config()
