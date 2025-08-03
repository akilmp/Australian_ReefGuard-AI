"""Kubeflow pipeline retraining the model when PSI exceeds a threshold."""
from __future__ import annotations

from kfp import dsl
from kfp.dsl import component

from .training_pipeline import training_pipeline


@component(base_image="python:3.10", packages_to_install=["pandas", "evidently"])
def compute_psi(reference_path: str, current_path: str) -> float:
    """Compute the Population Stability Index between two datasets."""
    import pandas as pd
    from evidently.report import Report
    from evidently.metrics import DataDriftTable

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    metrics = report.as_dict()["metrics"][0]["result"]
    return float(metrics.get("share_drifted_columns", 0.0))


@dsl.pipeline(
    name="retrain-on-drift",
    description="Compute PSI and run training pipeline if drift detected",
)
def retrain_on_drift(
    psi_threshold: float = 0.2,
    reference_path: str = "data/reference.csv",
    current_path: str = "data/current.csv",
):
    psi_task = compute_psi(reference_path=reference_path, current_path=current_path)
    with dsl.Condition(psi_task.output > psi_threshold):
        training_pipeline()


if __name__ == "__main__":
    from kfp.v2 import compiler

    compiler.Compiler().compile(retrain_on_drift, package_path="retrain_on_drift.json")
