"""Kubeflow pipeline retraining the model when PSI exceeds a threshold."""
from __future__ import annotations

from kfp import dsl
from kfp.dsl import component

from .training_pipeline import training_pipeline


@component(base_image="python:3.10", packages_to_install=["pandas", "numpy"])
def compute_psi(reference_path: str, current_path: str, bins: int = 10) -> float:
    """Compute the Population Stability Index between two datasets.

    The PSI is calculated for each numeric column and the average value is
    returned. Values of zero are replaced with a small constant to avoid
    division errors.
    """
    import pandas as pd
    import numpy as np

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    numeric_cols = [
        col
        for col in reference.columns
        if col in current.columns and pd.api.types.is_numeric_dtype(reference[col])
    ]
    if not numeric_cols:
        return 0.0

    def _psi(ref: pd.Series, cur: pd.Series) -> float:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.quantile(ref, quantiles)
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        ref_counts, bins_ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=bins_)
        ref_perc = ref_counts / len(ref)
        cur_perc = cur_counts / len(cur)
        ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
        cur_perc = np.where(cur_perc == 0, 1e-6, cur_perc)
        return np.sum((ref_perc - cur_perc) * np.log(ref_perc / cur_perc))

    psi_values = [_psi(reference[col].dropna(), current[col].dropna()) for col in numeric_cols]
    return float(np.mean(psi_values))


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
