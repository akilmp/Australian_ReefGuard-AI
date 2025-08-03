from unittest.mock import MagicMock

import pandas as pd

from models.inference.predictor import Predictor


def test_load_uses_model_dir_and_configures_feast(monkeypatch, tmp_path):
    """Ensure ``load`` pulls model from ``model_dir`` and sets up Feast."""
    load_model = MagicMock()
    monkeypatch.setattr(
        "mlflow.pyfunc.load_model", load_model
    )
    fake_fs = MagicMock()
    monkeypatch.setattr(
        "models.inference.predictor.FeatureStore", fake_fs
    )
    monkeypatch.setenv("FEAST_REPO_PATH", "/feast_repo")

    pred = Predictor("my-model", model_dir=str(tmp_path))
    pred.load()

    load_model.assert_called_once_with(str(tmp_path))
    fake_fs.assert_called_once_with(repo_path="/feast_repo")
    assert pred.model is load_model.return_value
    assert pred.feature_store is fake_fs.return_value
    assert pred.ready is True


def test_predict_fetches_features_and_runs_model(monkeypatch):
    """When only ``reef_id`` is provided, features are retrieved via Feast."""
    pred = Predictor("model")
    pred.model = MagicMock()
    pred.model.predict.return_value = pd.Series([0.1, 0.2])
    pred.ready = True

    feature_store = MagicMock()
    feature_store.get_online_features.return_value.to_df.return_value = pd.DataFrame(
        {
            "reef_id": [1, 2],
            "sst_turbidity_view:sst_celsius": [10, 20],
            "sst_turbidity_view:turbidity_ntu": [0.5, 0.7],
        }
    )
    pred.feature_store = feature_store

    request = {"instances": [{"reef_id": 1}, {"reef_id": 2}]}
    result = pred.predict(request)

    feature_store.get_online_features.assert_called_once()
    args, kwargs = pred.model.predict.call_args
    frame = args[0]
    assert list(frame.columns) == [
        "sst_turbidity_view:sst_celsius",
        "sst_turbidity_view:turbidity_ntu",
    ]
    assert result == {"predictions": [0.1, 0.2]}
