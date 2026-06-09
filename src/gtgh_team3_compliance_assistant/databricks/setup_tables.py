from __future__ import annotations
import argparse
from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark


def setup_tables(catalog: str, schema: str, volume: str) -> None:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume=volume)

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target.namespace}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {target.namespace}.{volume}")

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('bronze_raw_documents')} (
        document_id STRING,
        file_name STRING,
        source_path STRING,
        source_name STRING,
        regulation_family STRING,
        file_type STRING,
        ingestion_timestamp TIMESTAMP,
        status STRING
    )
    USING DELTA
    """)

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('silver_extracted_pages')} (
        document_id STRING,
        file_name STRING,
        page_number INT,
        page_text STRING,
        char_count INT,
        extraction_method STRING,
        extracted_at TIMESTAMP
    )
    USING DELTA
    """)

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('silver_chunks')} (
        chunk_uid STRING,
        document_id STRING,
        file_name STRING,
        chunk_id INT,
        page_number INT,
        chunk_type STRING,
        article_number STRING,
        annex_number STRING,
        title STRING,
        part_index INT,
        part_count INT,
        chunk_text STRING,
        char_length INT,
        law_passed_date STRING,
        ingested_at STRING,
        created_at TIMESTAMP
    )
    USING DELTA
    """)

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('gold_ai_ready_chunks')} (
        chunk_uid STRING,
        document_id STRING,
        file_name STRING,
        chunk_id INT,
        page_number INT,
        chunk_type STRING,
        article_number STRING,
        annex_number STRING,
        title STRING,
        part_index INT,
        part_count INT,
        chunk_text STRING,
        char_length INT,
        law_passed_date STRING,
        ingested_at STRING,
        source_path STRING,
        source_name STRING,
        regulation_family STRING,
        gold_created_at TIMESTAMP
    )
    USING DELTA
    """)

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('gold_embedded_chunks')} (
        chunk_uid STRING,
        document_id STRING,
        file_name STRING,
        chunk_id INT,
        page_number INT,
        chunk_type STRING,
        article_number STRING,
        annex_number STRING,
        title STRING,
        part_index INT,
        part_count INT,
        chunk_text STRING,
        char_length INT,
        law_passed_date STRING,
        ingested_at STRING,
        source_path STRING,
        source_name STRING,
        regulation_family STRING,
        embedding ARRAY<FLOAT>,
        embedding_model STRING,
        embedding_version STRING,
        embedded_at TIMESTAMP
    )
    USING DELTA
    """)

    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {target.table('pipeline_run_log')} (
        step_name STRING,
        status STRING,
        row_count BIGINT,
        message STRING,
        logged_at TIMESTAMP
    )
    USING DELTA
    """)

    print(f"Databricks target is ready: {target.namespace}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args()
    setup_tables(args.catalog, args.schema, args.volume)


if __name__ == "__main__":
    main()
