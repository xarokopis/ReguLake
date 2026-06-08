from __future__ import annotations

import argparse

from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark


def write_gold(catalog: str, schema: str) -> int:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume="unused")

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE {target.table('gold_ai_ready_chunks')} AS
        SELECT
            c.chunk_uid,
            c.document_id,
            c.file_name,
            CAST(c.chunk_id AS INT) AS chunk_id,
            CAST(c.page_number AS INT) AS page_number,
            c.chunk_type,
            c.article_number,
            c.annex_number,
            c.title,
            CAST(c.part_index AS INT) AS part_index,
            CAST(c.part_count AS INT) AS part_count,
            c.chunk_text,
            CAST(c.char_length AS INT) AS char_length,
            CAST(c.law_passed_date AS DATE) AS law_passed_date,
            CAST(c.ingested_at AS TIMESTAMP) AS ingested_at,
            d.source_path,
            d.source_name,
            d.regulation_family,
            current_timestamp() AS gold_created_at
        FROM {target.table('silver_chunks')} c
        LEFT JOIN {target.table('bronze_raw_documents')} d
            ON c.document_id = d.document_id
        """
    )

    count = spark.table(target.table("gold_ai_ready_chunks")).count()
    print(f"Wrote {count} rows to {target.table('gold_ai_ready_chunks')}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    write_gold(args.catalog, args.schema)


if __name__ == "__main__":
    main()
