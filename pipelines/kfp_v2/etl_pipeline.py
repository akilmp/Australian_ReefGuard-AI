#!/usr/bin/env python
"""Kubeflow v2 pipeline pulling Sentinel-2, MODIS SST and QLD buoy data.

Each component fetches external data and stages it into lakeFS-backed
S3 storage so downstream steps can access raw artefacts.
"""

from kfp import dsl
from kfp.dsl import component


@component(base_image="python:3.10", packages_to_install=["requests", "boto3"])
def pull_sentinel2(
    lakefs_endpoint: str,
    bucket: str,
    prefix: str,
    region: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch a Sentinel-2 COG tile and upload to lakeFS."""
    import os
    import requests
    import boto3

    stac_url = "https://earth-search.aws.element84.com/v1/search"
    query = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [float(x) for x in region.split(",")],
        "datetime": f"{start_date}/{end_date}",
        "limit": 1,
    }
    response = requests.post(stac_url, json=query, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError("No Sentinel-2 items found for query")
    asset_url = features[0]["assets"]["visual"]["href"]

    data = requests.get(asset_url, timeout=60).content

    s3 = boto3.client("s3", endpoint_url=lakefs_endpoint)
    key = os.path.join(prefix, "sentinel2_visual.tif")
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


@component(base_image="python:3.10", packages_to_install=["requests", "boto3"])
def pull_modis_sst(
    lakefs_endpoint: str,
    bucket: str,
    prefix: str,
    date: str,
) -> str:
    """Fetch a MODIS sea-surface-temperature file and upload to lakeFS."""
    import os
    import requests
    import boto3

    url = (
        "https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/"
        f"{date}/AQUA_MODIS.{date}.L3m.DAY.SST.sst.4km.nc"
    )
    data = requests.get(url, timeout=60).content

    s3 = boto3.client("s3", endpoint_url=lakefs_endpoint)
    key = os.path.join(prefix, f"modis_sst_{date}.nc")
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


@component(base_image="python:3.10", packages_to_install=["requests", "boto3"])
def pull_qld_buoy(
    lakefs_endpoint: str,
    bucket: str,
    prefix: str,
    station_id: str,
    api_key: str | None = None,
    base_url: str = "https://weather.aims.gov.au/api/v1/stations",
) -> str:
    """Fetch Queensland buoy JSON telemetry and upload to lakeFS.

    Parameters
    ----------
    lakefs_endpoint: str
        lakeFS endpoint URL for S3-compatible uploads.
    bucket: str
        Target lakeFS bucket.
    prefix: str
        Key prefix inside the bucket.
    station_id: str
        Identifier for the Queensland IMOS/AIMS buoy station.
    api_key: str | None, optional
        Optional API key for the AIMS Weather API.  If provided the
        ``x-api-key`` header will be sent with the request.
    base_url: str, optional
        Base URL for the AIMS Weather API station endpoint.
    """

    import os
    import requests
    import boto3

    headers = {"x-api-key": api_key} if api_key else {}
    params = {"format": "json"}
    url = f"{base_url}/{station_id}"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch buoy data from {url}"
        ) from exc

    data = response.content

    s3 = boto3.client("s3", endpoint_url=lakefs_endpoint)
    key = os.path.join(prefix, f"buoy_{station_id}.json")
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


@dsl.pipeline(name="reefguard-etl", description="Stage remote datasets into lakeFS")
def etl_pipeline(
    lakefs_endpoint: str,
    bucket: str,
    prefix: str = "raw",
    region: str = "146,-19,147,-18",
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-02",
    modis_date: str = "2023-001",
    buoy_station: str = "station-001",
    buoy_api_key: str = "",
):
    pull_sentinel2(
        lakefs_endpoint=lakefs_endpoint,
        bucket=bucket,
        prefix=prefix,
        region=region,
        start_date=start_date,
        end_date=end_date,
    )
    pull_modis_sst(
        lakefs_endpoint=lakefs_endpoint,
        bucket=bucket,
        prefix=prefix,
        date=modis_date,
    )
    pull_qld_buoy(
        lakefs_endpoint=lakefs_endpoint,
        bucket=bucket,
        prefix=prefix,
        station_id=buoy_station,
        api_key=buoy_api_key or None,
    )


if __name__ == "__main__":
    from kfp.v2 import compiler

    compiler.Compiler().compile(etl_pipeline, package_path="etl_pipeline.json")
