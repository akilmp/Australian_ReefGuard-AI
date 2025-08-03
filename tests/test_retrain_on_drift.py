import importlib
import sys
import types


class _SkipBlock(Exception):
    pass


class _FakeCondition:
    def __init__(self, predicate):
        self.predicate = predicate

    def __enter__(self):
        if not self.predicate:
            raise _SkipBlock()

    def __exit__(self, exc_type, exc, tb):
        return exc_type is _SkipBlock


def _run_pipeline(monkeypatch, psi_value):
    kfp_stub = types.ModuleType("kfp")
    dsl_stub = types.ModuleType("kfp.dsl")

    def _identity_decorator(*args, **kwargs):
        def _wrap(func):
            return func

        return _wrap

    dsl_stub.component = _identity_decorator
    dsl_stub.pipeline = _identity_decorator
    dsl_stub.container_component = _identity_decorator
    dsl_stub.Condition = _FakeCondition
    kfp_stub.dsl = dsl_stub
    monkeypatch.setitem(sys.modules, "kfp", kfp_stub)
    monkeypatch.setitem(sys.modules, "kfp.dsl", dsl_stub)
    module_name = "pipelines.kfp_v2.retrain_on_drift"
    retrain_on_drift = importlib.import_module(module_name)

    class _FakeTask:
        output = psi_value

    def _fake_compute_psi(**_):
        return _FakeTask()

    monkeypatch.setattr(
        retrain_on_drift,
        "compute_psi",
        _fake_compute_psi,
    )
    called = {"flag": False}

    def _fake_training_pipeline():
        called["flag"] = True

    monkeypatch.setattr(
        retrain_on_drift,
        "training_pipeline",
        _fake_training_pipeline,
    )

    try:
        retrain_on_drift.retrain_on_drift(psi_threshold=0.2)
    except _SkipBlock:
        pass
    return called["flag"]


def test_triggers_training_when_drift(monkeypatch):
    assert _run_pipeline(monkeypatch, 0.3)


def test_skips_training_without_drift(monkeypatch):
    assert not _run_pipeline(monkeypatch, 0.1)
