"""Periodic drift evaluation job triggering retraining when PSI exceeds threshold."""
import os
import pandas as pd
import requests
from evidently.report import Report
from evidently.metrics import DataDriftTable

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
RETRAIN_ENDPOINT = os.getenv("RETRAIN_ENDPOINT", "http://retrainer/retrain")


def load_data():
    """Load reference and current datasets.

    This function expects CSV files `data/reference.csv` and
    `data/current.csv` to be mounted into the container. In a real
    deployment these would come from a data lake or feature store.
    """
    reference = pd.read_csv("data/reference.csv")
    current = pd.read_csv("data/current.csv")
    return reference, current


def compute_psi(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Compute the Population Stability Index using Evidently."""
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    metrics = report.as_dict()["metrics"][0]["result"]
    # DataDriftTable returns share of drifted columns; treat as PSI
    return metrics.get("share_drifted_columns", 0.0)


def trigger_retraining():
    """Invoke retraining pipeline via REST endpoint."""
    response = requests.post(RETRAIN_ENDPOINT, timeout=10)
    response.raise_for_status()


def main() -> None:
    reference, current = load_data()
    psi = compute_psi(reference, current)
    if psi > PSI_THRESHOLD:
        trigger_retraining()


if __name__ == "__main__":
    main()
