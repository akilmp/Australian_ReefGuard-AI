import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fake_kfp(monkeypatch):
    """Provide a minimal kfp stub so the pipeline module imports."""
    kfp_module = types.ModuleType("kfp")
    dsl_module = types.ModuleType("dsl")

    def identity_decorator(*args, **kwargs):
        def wrap(func):
            return func
        return wrap

    dsl_module.component = identity_decorator
    dsl_module.pipeline = identity_decorator
    kfp_module.dsl = dsl_module
    monkeypatch.setitem(sys.modules, "kfp", kfp_module)
    monkeypatch.setitem(sys.modules, "kfp.dsl", dsl_module)


def test_pull_sentinel2_uploads_to_s3(monkeypatch):
    post_resp = MagicMock()
    post_resp.json.return_value = {
        "features": [{"assets": {"visual": {"href": "http://asset"}}}]
    }
    post_resp.raise_for_status.return_value = None
    get_resp = MagicMock()
    get_resp.content = b"data"

    requests_mod = types.ModuleType("requests")
    requests_mod.post = MagicMock(return_value=post_resp)
    requests_mod.get = MagicMock(return_value=get_resp)
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    s3_client = MagicMock()
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = MagicMock(return_value=s3_client)
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)

    etl = importlib.import_module("pipelines.kfp_v2.etl_pipeline")
    uri = etl.pull_sentinel2(
        lakefs_endpoint="http://lf",
        bucket="bucket",
        prefix="prefix",
        region="1,2,3,4",
        start_date="2023-01-01",
        end_date="2023-01-02",
    )

    assert uri == "s3://bucket/prefix/sentinel2_visual.tif"
    boto3_mod.client.assert_called_once_with("s3", endpoint_url="http://lf")
    s3_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="prefix/sentinel2_visual.tif", Body=b"data"
    )


def test_pull_modis_sst(monkeypatch):
    get_resp = MagicMock()
    get_resp.content = b"sst"
    requests_mod = types.ModuleType("requests")
    requests_mod.get = MagicMock(return_value=get_resp)
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    s3_client = MagicMock()
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = MagicMock(return_value=s3_client)
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)

    etl = importlib.import_module("pipelines.kfp_v2.etl_pipeline")
    uri = etl.pull_modis_sst(
        lakefs_endpoint="http://lf",
        bucket="bucket",
        prefix="prefix",
        date="2023-001",
    )

    assert uri == "s3://bucket/prefix/modis_sst_2023-001.nc"
    requests_mod.get.assert_called_once()
    s3_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="prefix/modis_sst_2023-001.nc", Body=b"sst"
    )


def test_pull_qld_buoy(monkeypatch):
    get_resp = MagicMock()
    get_resp.content = b"{}"
    get_resp.raise_for_status.return_value = None
    requests_mod = types.ModuleType("requests")
    requests_mod.get = MagicMock(return_value=get_resp)
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    s3_client = MagicMock()
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = MagicMock(return_value=s3_client)
    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)

    etl = importlib.import_module("pipelines.kfp_v2.etl_pipeline")
    uri = etl.pull_qld_buoy(
        lakefs_endpoint="http://lf",
        bucket="bucket",
        prefix="pref",
        station_id="station-001",
        api_key="KEY",
    )

    assert uri == "s3://bucket/pref/buoy_station-001.json"
    requests_mod.get.assert_called_once()
    args, kwargs = requests_mod.get.call_args
    assert kwargs["headers"] == {"x-api-key": "KEY"}
    assert kwargs["params"] == {"format": "json"}
    s3_client.put_object.assert_called_once_with(
        Bucket="bucket", Key="pref/buoy_station-001.json", Body=b"{}"
    )
