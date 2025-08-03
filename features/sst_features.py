from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32
from feast.file_format import ParquetFormat

# Entity representing a reef location
reef_id = Entity(name="reef_id", join_keys=["reef_id"])

# Offline data source stored as Parquet files on S3
sst_turbidity_source = FileSource(
    name="sst_turbidity_source",
    path="s3://reefguard/features/sst_turbidity.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    file_format=ParquetFormat(),
)

# Feature view exposing sea surface temperature and turbidity features
sst_turbidity_view = FeatureView(
    name="sst_turbidity_view",
    entities=[reef_id],
    ttl=timedelta(hours=24),
    schema=[
        Field(name="sst_celsius", dtype=Float32),
        Field(name="turbidity_ntu", dtype=Float32),
    ],
    online=True,
    source=sst_turbidity_source,
    tags={"team": "reefguard"},
)
