import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def load_train_module(monkeypatch):
    """Import training module with a stubbed mlflow package."""
    logged: dict[str, float | bool] = {}

    mlflow_module = types.ModuleType("mlflow")

    class DummyRun:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    mlflow_module.start_run = lambda: DummyRun()
    mlflow_module.log_metric = lambda k, v: logged.setdefault(k, v)

    class SklearnModule:
        @staticmethod
        def log_model(model, artifact_path):
            logged["model_logged"] = True

    sklearn_module = SklearnModule()
    mlflow_module.sklearn = sklearn_module
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", sklearn_module)
    mlflow_module.get_artifact_uri = lambda path: f"/tmp/{path}"

    class DummyResult:
        name = "model"
        version = 1

    mlflow_module.register_model = lambda uri, name: DummyResult()

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)

    spec = importlib.util.spec_from_file_location(
        "train", Path("models/trainer/train.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, logged


def test_train_runs(monkeypatch):
    module, logged = load_train_module(monkeypatch)
    args = SimpleNamespace(model_name="model", experiment="exp")
    module.train(args)
    assert "accuracy" in logged
    assert logged["model_logged"] is True
