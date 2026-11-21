"""Transaction scoring API endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger

router = APIRouter()


# --- Request / Response Schemas ---

class TransactionRequest(BaseModel):
    transaction_id: str
    timestamp: datetime
    portfolio_id: str
    asset_class: str
    transaction_type: str
    transaction_amount: float = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    counterparty_id: str


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    anomaly_score: float
    is_anomaly: bool
    risk_score: float
    risk_label: str
    scored_at: datetime


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchScoreResponse(BaseModel):
    results: list[TransactionScoreResponse]
    total: int
    anomaly_count: int
    high_risk_count: int


# --- Dependency: lazy-load inference pipeline ---

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from src.pipelines.inference_pipeline import InferencePipeline
            _pipeline = InferencePipeline()
        except Exception as e:
            logger.error(f"Failed to load inference pipeline: {e}")
            raise HTTPException(status_code=503, detail="Models not available.")
    return _pipeline


# --- Endpoints ---

@router.post("/score", response_model=TransactionScoreResponse)
async def score_transaction(
    request: TransactionRequest,
    pipeline=Depends(get_pipeline),
):
    """Score a single transaction for anomaly and risk."""
    result = pipeline.score_single(request.model_dump())
    return TransactionScoreResponse(
        transaction_id=request.transaction_id,
        anomaly_score=float(result.get("anomaly_score", 0)),
        is_anomaly=bool(result.get("is_anomaly", False)),
        risk_score=float(result.get("risk_score", 0)),
        risk_label=str(result.get("risk_label", "UNKNOWN")),
        scored_at=datetime.utcnow(),
    )


@router.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch(
    request: BatchTransactionRequest,
    pipeline=Depends(get_pipeline),
):
    """Score a batch of transactions."""
    df = pd.DataFrame([t.model_dump() for t in request.transactions])
    df_scored = pipeline.score(df)

    results = []
    for _, row in df_scored.iterrows():
        results.append(TransactionScoreResponse(
            transaction_id=row["transaction_id"],
            anomaly_score=float(row.get("anomaly_score", 0)),
            is_anomaly=bool(row.get("is_anomaly", False)),
            risk_score=float(row.get("risk_score", 0)),
            risk_label=str(row.get("risk_label", "UNKNOWN")),
            scored_at=datetime.utcnow(),
        ))

    return BatchScoreResponse(
        results=results,
        total=len(results),
        anomaly_count=sum(1 for r in results if r.is_anomaly),
        high_risk_count=sum(1 for r in results if r.risk_label == "HIGH"),
    )
