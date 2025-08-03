import importlib.util
import sys
import types
from pathlib import Path

import pytest
import requests


def load_etl_module(monkeypatch):
    """Import the ETL pipeline module with stubbed kfp dependency."""
    kfp_module = types.ModuleType("kfp")
    dsl_module = types.ModuleType("dsl")

    def _passthrough_decorator(*_args, **_kwargs):
        def wrapper(func):
            return func
        return wrapper

    dsl_module.component = _passthrough_decorator
    dsl_module.pipeline = _passthrough_decorator
    kfp_module.dsl = dsl_module

    monkeypatch.setitem(sys.modules, "kfp", kfp_module)
    monkeypatch.setitem(sys.modules, "kfp.dsl", dsl_module)

    spec = importlib.util.spec_from_file_location(
        "etl_pipeline", Path("pipelines/kfp_v2/etl_pipeline.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pull_sentinel2(monkeypatch):
    etl = load_etl_module(monkeypatch)

    class MockPost:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "features": [
                    {"assets": {"visual": {"href": "http://example.com/vis.tif"}}}
                ]
            }

    class MockGet:
        content = b"image"

    monkeypatch.setattr(requests, "post", lambda *a, **k: MockPost())
    monkeypatch.setattr(requests, "get", lambda *a, **k: MockGet())

    class DummyS3:
        def __init__(self):
            self.calls = []

        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    s3_client = DummyS3()

    def client(service, endpoint_url=None):
        assert service == "s3"
        return s3_client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))

    result = etl.pull_sentinel2(
        lakefs_endpoint="http://lakefs",
        bucket="bucket",
        prefix="prefix",
        region="0,0,1,1",
        start_date="2023-01-01",
        end_date="2023-01-02",
    )

    assert result == "s3://bucket/prefix/sentinel2_visual.tif"


def test_pull_modis_sst(monkeypatch):
    etl = load_etl_module(monkeypatch)

    class MockGet:
        content = b"sst"

    monkeypatch.setattr(requests, "get", lambda *a, **k: MockGet())

    class DummyS3:
        def __init__(self):
            self.calls = []

        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    s3_client = DummyS3()

    def client(service, endpoint_url=None):
        assert service == "s3"
        return s3_client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))

    result = etl.pull_modis_sst(
        lakefs_endpoint="http://lakefs",
        bucket="bucket",
        prefix="prefix",
        date="2023-001",
    )

    assert result == "s3://bucket/prefix/modis_sst_2023-001.nc"


def test_pull_qld_buoy(monkeypatch):
    etl = load_etl_module(monkeypatch)

    class MockGet:
        content = b"buoy"

    monkeypatch.setattr(requests, "get", lambda *a, **k: MockGet())

    class DummyS3:
        def __init__(self):
            self.calls = []

        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    s3_client = DummyS3()

    def client(service, endpoint_url=None):
        assert service == "s3"
        return s3_client

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))

    result = etl.pull_qld_buoy(
        lakefs_endpoint="http://lakefs",
        bucket="bucket",
        prefix="prefix",
        station_id="station-001",
    )

    assert result == "s3://bucket/prefix/buoy_station-001.json"
