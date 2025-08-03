import importlib.util
import sys
import types
from pathlib import Path


def load_predictor(monkeypatch):
    """Import predictor module with a stubbed kserve package."""
    kserve_module = types.ModuleType("kserve")

    class KFModel:
        def __init__(self, name):
            self.name = name

    class ModelServer:
        def start(self, models):
            return None

    kserve_module.KFModel = KFModel
    kserve_module.ModelServer = ModelServer

    monkeypatch.setitem(sys.modules, "kserve", kserve_module)
    spec = importlib.util.spec_from_file_location(
        "predictor", Path("models/inference/predictor.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_predictor(monkeypatch):
    module = load_predictor(monkeypatch)
    predictor = module.Predictor("reefguard-model")
    assert not predictor.ready
    predictor.load()
    assert predictor.ready
    result = predictor.predict({"instances": [1, 2, 3]})
    assert result == {"predictions": [1, 2, 3]}
