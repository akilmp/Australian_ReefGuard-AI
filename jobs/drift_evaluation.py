"""Periodic job submitting retrain-on-drift pipeline."""
import os
from kfp import Client

from pipelines.kfp_v2.retrain_on_drift import retrain_on_drift

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
KFP_HOST = os.getenv("KFP_HOST", "http://ml-pipeline.kubeflow.svc.cluster.local:8888")


def main() -> None:
    """Submit the retraining pipeline with the configured PSI threshold."""
    client = Client(host=KFP_HOST)
    client.create_run_from_pipeline_func(
        retrain_on_drift,
        arguments={"psi_threshold": PSI_THRESHOLD},
    )


if __name__ == "__main__":
    main()
