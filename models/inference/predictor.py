"""Predictor implementation for KFServing.

This module loads an MLflow‑registered model and serves predictions via
``kserve``. If a Feast feature repository path is supplied through the
``FEAST_REPO_PATH`` environment variable, online features are retrieved for
incoming ``reef_id`` values before inference.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List

import kserve
import mlflow.pyfunc
import pandas as pd

try:  # Feast is optional at runtime
    from feast import FeatureStore
except Exception:  # pragma: no cover - Feast may not be installed
    FeatureStore = None


class Predictor(kserve.KFModel):
    """Model predictor for KFServing.

    During :meth:`load`, an MLflow model is loaded either from the supplied
    ``model_dir`` or from the MLflow Model Registry. If a Feast repository is
    configured, online features are fetched for the provided entities prior to
    calling the model's ``predict`` function.
    """

    def __init__(self, name: str, model_dir: str | None = None):
        super().__init__(name)
        self.model_dir = model_dir
        self.ready = False
        self.model: mlflow.pyfunc.PyFuncModel | None = None
        self.feature_store: FeatureStore | None = None

    def load(self) -> None:
        """Load the MLflow model and optionally configure Feast."""

        # Determine model URI: explicit ``model_dir`` takes precedence, else
        # load from the Model Registry using the provided name and stage.
        if self.model_dir:
            model_uri = self.model_dir
        else:
            model_name = os.getenv("MLFLOW_MODEL_NAME", self.name)
            model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
            model_uri = f"models:/{model_name}/{model_stage}"

        self.model = mlflow.pyfunc.load_model(model_uri)

        # Configure Feast feature store if requested
        feast_repo = os.getenv("FEAST_REPO_PATH")
        if FeatureStore is not None and feast_repo:
            self.feature_store = FeatureStore(repo_path=feast_repo)

        self.ready = True

    def predict(self, request: Dict) -> Dict[str, List]:
        """Generate predictions from the request payload.

        Parameters
        ----------
        request:
            The request payload passed by KFServing. The JSON object should
            contain an ``instances`` field with a list of inputs. Each
            instance may either include all required feature values or just a
            ``reef_id`` field, in which case Feast will supply the features.

        Returns
        -------
        Dict[str, List]
            A dictionary with a ``predictions`` key containing the model
            outputs.
        """

        if not self.ready:
            logging.error("Predict called before model was ready")
            raise RuntimeError("Predictor is not ready. Call 'load' before 'predict'.")
        if not self.model:
            logging.error("Predict called without a loaded model")
            raise RuntimeError("Model is not loaded")

        instances = request.get("instances")
        if (
            not isinstance(instances, list)
            or not instances
            or not all(isinstance(inst, dict) for inst in instances)
        ):
            logging.error("Invalid 'instances' payload: %s", instances)
            raise ValueError(
                "request['instances'] must be a non-empty list of dictionaries"
            )

        if self.feature_store and instances and isinstance(instances[0], dict):
            # Retrieve features for each reef_id if explicit feature values are
            # absent in the request
            needs_lookup = any(
                "reef_id" in inst and (len(inst.keys()) == 1) for inst in instances
            )
            if needs_lookup:
                entity_rows = [{"reef_id": inst["reef_id"]} for inst in instances]
                feature_vector = self.feature_store.get_online_features(
                    features=[
                        "sst_turbidity_view:sst_celsius",
                        "sst_turbidity_view:turbidity_ntu",
                    ],
                    entity_rows=entity_rows,
                ).to_df()
                # Drop entity column to obtain pure feature matrix
                data = feature_vector.drop(columns=["reef_id"]).to_dict(orient="records")
                instances = data

        # Convert instances to a DataFrame and run model prediction
        frame = pd.DataFrame(instances)
        predictions = self.model.predict(frame)

        return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    model = Predictor("reefguard-model")
    model.load()
    kserve.ModelServer().start([model])
