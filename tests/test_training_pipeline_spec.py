import importlib
import json
import sys
import types


def test_katib_experiment_spec(monkeypatch):
    kfp_stub = types.ModuleType("kfp")

    def _identity_decorator(*args, **kwargs):
        def _wrap(func):
            return func

        return _wrap

    kfp_stub.dsl = types.SimpleNamespace(
        pipeline=_identity_decorator,
        container_component=_identity_decorator,
        ContainerSpec=object,
    )
    monkeypatch.setitem(sys.modules, "kfp", kfp_stub)
    module_name = "pipelines.kfp_v2.training_pipeline"
    training_pipeline = importlib.import_module(module_name)

    captured = {}

    def fake_katib_experiment(experiment_spec: str):
        captured["spec"] = experiment_spec

    monkeypatch.setattr(
        training_pipeline,
        "katib_experiment",
        fake_katib_experiment,
    )

    training_pipeline.training_pipeline()

    spec = json.loads(captured["spec"])
    param_names = {p["name"] for p in spec["spec"]["parameters"]}
    assert param_names == {
        "xgbLearningRate",
        "xgbMaxDepth",
        "xgbNEstimators",
        "vitLr",
        "vitEpochs",
        "vitBatchSize",
    }
    objective_metric_name = spec["spec"]["objective"]["objectiveMetricName"]
    assert objective_metric_name == "ensemble_accuracy"
