import os
from pathlib import Path

import mlflow


APPROVED_FILE = Path("models/approved_models.txt")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    approved = {
        line.strip() for line in APPROVED_FILE.read_text().splitlines() if line.strip()
    }
    client = mlflow.MlflowClient()

    for rm in client.list_registered_models():
        name = rm.name
        latest = client.get_latest_versions(name)
        for mv in latest:
            if mv.current_stage in {"Staging", "Production"} and name not in approved:
                msg = f"Model {name} in stage {mv.current_stage} is not approved"
                raise SystemExit(msg)
    print("All models in staging or production are approved")


if __name__ == "__main__":
    main()
