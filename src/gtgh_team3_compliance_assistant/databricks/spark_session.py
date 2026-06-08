from __future__ import annotations
from functools import lru_cache

@lru_cache(maxsize=1)
def get_spark():
    """Return a Spark session backed by Databricks serverless compute."""
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.serverless(True).getOrCreate()