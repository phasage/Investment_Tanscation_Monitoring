# Investment Transaction Monitoring

An AI/ML system for monitoring investment transactions, detecting anomalies, flagging suspicious activity, and generating risk insights in real time.

## Project Overview

| Capability | Description |
|---|---|
| Anomaly Detection | Identify unusual transaction patterns using unsupervised ML |
| Risk Scoring | Assign real-time risk scores to transactions |
| Fraud Detection | Classify potentially fraudulent activity |
| Trend Analysis | Surface portfolio and market trends |
| Alerting | Generate alerts for compliance and operations teams |

## Project Structure

```
├── data/               # Raw, processed, and feature datasets
├── notebooks/          # EDA, modeling, and evaluation notebooks
├── src/                # Source code (data, features, models, API)
├── tests/              # Unit and integration tests
├── configs/            # Configuration files (YAML)
├── models/             # Trained model artifacts
├── reports/            # Figures and metric reports
├── scripts/            # Utility and automation scripts
├── docker/             # Containerization files
└── logs/               # Runtime logs
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Run the pipeline
python scripts/run_pipeline.py

# 5. Start the API
uvicorn src.api.main:app --reload
```

## Setup & Installation

See [docs](#) for full setup, data ingestion, and deployment instructions.

## Tech Stack

- **ML**: scikit-learn, XGBoost, PyTorch, Isolation Forest
- **Data**: pandas, polars, SQLAlchemy, Apache Kafka
- **API**: FastAPI, Pydantic
- **Monitoring**: MLflow, Evidently AI, Prometheus
- **Infra**: Docker, GitHub Actions
