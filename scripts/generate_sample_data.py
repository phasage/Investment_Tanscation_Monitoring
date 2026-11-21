"""Generate synthetic transaction data for development and testing."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import click


@click.command()
@click.option("--n", default=10_000, help="Number of transactions to generate")
@click.option("--out", default="data/raw/transactions.csv", help="Output path")
@click.option("--anomaly-rate", default=0.03, help="Fraction of anomalous transactions")
@click.option("--seed", default=42, help="Random seed")
def generate(n: int, out: str, anomaly_rate: float, seed: int):
    """Generate synthetic investment transaction data."""
    rng = np.random.default_rng(seed)

    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=int(x)) for x in
                  np.cumsum(rng.exponential(scale=5, size=n))]

    portfolios = [f"PORT_{i:03d}" for i in range(1, 21)]
    asset_classes = ["equity", "bond", "fx", "commodity", "derivative"]
    tx_types = ["BUY", "SELL", "TRANSFER", "DIVIDEND", "REBALANCE"]
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]
    counterparties = [f"CP_{i:03d}" for i in range(1, 51)]

    amounts = rng.lognormal(mean=10, sigma=1.5, size=n)

    # Inject anomalies
    n_anomalies = int(n * anomaly_rate)
    anomaly_idx = rng.choice(n, n_anomalies, replace=False)
    amounts[anomaly_idx] *= rng.uniform(10, 100, n_anomalies)
    is_high_risk = np.zeros(n, dtype=int)
    is_high_risk[anomaly_idx] = 1

    df = pd.DataFrame({
        "transaction_id": [f"TX{i:08d}" for i in range(n)],
        "timestamp": timestamps,
        "portfolio_id": rng.choice(portfolios, n),
        "asset_class": rng.choice(asset_classes, n),
        "transaction_type": rng.choice(tx_types, n),
        "transaction_amount": amounts.round(2),
        "currency": rng.choice(currencies, n),
        "counterparty_id": rng.choice(counterparties, n),
        "is_high_risk": is_high_risk,
    })

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    click.echo(f"Generated {n:,} transactions → {out_path}")
    click.echo(f"  High-risk (labelled): {is_high_risk.sum():,} ({anomaly_rate:.1%})")


if __name__ == "__main__":
    generate()
