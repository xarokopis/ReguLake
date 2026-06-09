from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone

from pyspark.sql import Row
from pyspark.sql.functions import col, current_timestamp, to_date, to_timestamp
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark
from gtgh_team3_compliance_assistant.processing.eur_chunker import EurChunker


LAW_PASSED_DATES = {
    "32014L0065_EN": "2014-05-15",
    "32015L2366_EN": "2015-11-25",
    "32016R0679_EN": "2016-04-27",
    "32024L1640_EN": "2024-05-31",
    "32024R1689_EN": "2024-06-13",
}


CHUNKS_SCHEMA = StructType(
    [
        StructField("chunk_uid", StringType(), False),
        StructField("document_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("chunk_id", IntegerType(), False),
        StructField("page_number", IntegerType(), True),
        StructField("chunk_type", StringType(), True),
        StructField("article_number", StringType(), True),
        StructField("annex_number", StringType(), True),
        StructField("title", StringType(), True),
        StructField("part_index", IntegerType(), False),
        StructField("part_count", IntegerType(), False),
        StructField("chunk_text", StringType(), True),
        StructField("char_length", IntegerType(), False),
        StructField("law_passed_date", StringType(), True),
        StructField("ingested_at", StringType(), False),
    ]
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower().strip()


def _find_page(chunk: dict, normalized_pages: list[str]) -> int | None:
    text = chunk["text"]
    part_index = chunk.get("part_index", 0)

    if part_index > 0 and chunk.get("article_number"):
        match = re.search(r"(?:\(\w+\)|\d+\.)\s", text)
        if match:
            text = text[match.start() :]
    elif chunk.get("type") == "article" and chunk.get("article_number"):
        target = f"Article {chunk['article_number']}"
        idx = text.find(target)
        if idx > 0:
            text = text[idx:]
    elif chunk.get("type") == "annex" and chunk.get("annex_number"):
        target = f"ANNEX {chunk['annex_number']}"
        idx = text.find(target)
        if idx > 0:
            text = text[idx:]

    normalized_text = _normalize(text)
    if not normalized_text:
        return None

    for fp_len in (200, 100, 50):
        fingerprint = normalized_text[:fp_len]
        if not fingerprint:
            continue
        for page_num, normalized_page in enumerate(normalized_pages, start=1):
            if fingerprint in normalized_page:
                return int(page_num)

    return None


def _build_uid(document_id: str, chunk: dict, idx: int) -> str:
    chunk_type = chunk.get("type")
    part_index = chunk.get("part_index", 0)

    if chunk_type == "article" and chunk.get("article_number"):
        return f"{document_id}_art_{chunk['article_number']}_p_{part_index}"

    if chunk_type == "annex" and chunk.get("annex_number"):
        return f"{document_id}_ann_{chunk['annex_number']}_p_{part_index}"

    return f"{document_id}_chunk_{idx}"


def write_chunks(catalog: str, schema: str) -> int:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume="unused")
    chunker = EurChunker()

    pages = (
        spark.table(target.table("silver_extracted_pages"))
        .orderBy("document_id", "page_number")
        .collect()
    )

    by_doc: dict[tuple[str, str], list[str]] = {}
    for page in pages:
        key = (page["document_id"], page["file_name"])
        by_doc.setdefault(key, []).append(page["page_text"])

    rows = []
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for (document_id, file_name), page_texts in by_doc.items():
        full_text = "\n".join(page_texts)
        raw_chunks = chunker.chunk(full_text)
        normalized_pages = [_normalize(page) for page in page_texts]

        for idx, chunk in enumerate(raw_chunks):
            chunk_text = chunk["text"]
            rows.append(
                Row(
                    chunk_uid=_build_uid(document_id, chunk, idx),
                    document_id=document_id,
                    file_name=file_name,
                    chunk_id=int(idx),
                    page_number=_find_page(chunk, normalized_pages),
                    chunk_type=chunk.get("type"),
                    article_number=chunk.get("article_number"),
                    annex_number=chunk.get("annex_number"),
                    title=chunk.get("title"),
                    part_index=int(chunk.get("part_index", 0) or 0),
                    part_count=int(chunk.get("part_count", 1) or 1),
                    chunk_text=chunk_text,
                    char_length=int(len(chunk_text)),
                    law_passed_date=LAW_PASSED_DATES.get(document_id),
                    ingested_at=ingested_at,
                )
            )

    if not rows:
        raise ValueError("No extracted pages were found to chunk")

    df = (
        spark.createDataFrame(rows, schema=CHUNKS_SCHEMA)
        .withColumn("chunk_id", col("chunk_id").cast("int"))
        .withColumn("page_number", col("page_number").cast("int"))
        .withColumn("part_index", col("part_index").cast("int"))
        .withColumn("part_count", col("part_count").cast("int"))
        .withColumn("char_length", col("char_length").cast("int"))
        .withColumn("law_passed_date", to_date(col("law_passed_date")))
        .withColumn("ingested_at", to_timestamp(col("ingested_at")))
        .withColumn("created_at", current_timestamp())
    )

    table_name = target.table("silver_chunks")
    (
        df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .format("delta")
        .saveAsTable(table_name)
    )

    count = df.count()
    print(f"Wrote {count} rows to {table_name}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    write_chunks(args.catalog, args.schema)


if __name__ == "__main__":
    main()
