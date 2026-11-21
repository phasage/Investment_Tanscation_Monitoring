"""Centralized logging setup using loguru."""
import sys
from pathlib import Path
from loguru import logger

from src.utils.config import get_settings

settings = get_settings()

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger():
    """Configure loguru logger for the application."""
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler with rotation
    logger.add(
        LOG_DIR / "app.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Separate error log
    logger.add(
        LOG_DIR / "error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="50 MB",
        retention="90 days",
    )

    return logger


app_logger = setup_logger()
