# Investment Transaction Monitoring — Project Steering

## Domain
Financial AI/ML system for real-time investment transaction monitoring, anomaly detection, risk scoring, and compliance alerting.

## Architecture Principles
- **Separation of concerns**: data ingestion, feature engineering, model training, inference, and API are distinct modules under `src/`.
- **Config-driven**: all thresholds, model hyperparameters, and paths come from `configs/config.yaml` or environment variables via `.env`.
- **Reproducibility**: all experiments tracked in MLflow; datasets versioned separately (never committed to git).
- **Fail-safe inference**: the API degrades gracefully if models are unavailable (503, not 500).

## Key Directories
| Path | Purpose |
|---|---|
| `src/data/` | Data ingestion and validation |
| `src/features/` | Feature engineering |
| `src/models/` | ML model classes (anomaly, risk, fraud) |
| `src/pipelines/` | Training and inference orchestration |
| `src/api/` | FastAPI REST service |
| `src/monitoring/` | Drift and performance monitoring |
| `configs/` | YAML configuration |
| `scripts/` | CLI tools for data generation and pipeline runs |

## Coding Standards
- Python 3.11+, type hints on all public functions
- `loguru` for logging (not `print`)
- Pydantic v2 for all API schemas
- pytest for all tests; aim for >80% coverage on `src/`
- Format with `black`, lint with `ruff`

## Data Sensitivity
- Never commit raw transaction data (`data/raw/`, `data/processed/`) — these are in `.gitignore`
- Never hardcode credentials — use `.env` (not committed) or environment variables
- Use parameterized queries for all database interactions
