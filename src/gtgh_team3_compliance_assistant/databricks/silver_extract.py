from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql.functions import col, current_timestamp
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark
from gtgh_team3_compliance_assistant.processing.text_extractor import TextExtractor


EXTRACTED_PAGES_SCHEMA = StructType(
    [
        StructField("document_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("page_number", IntegerType(), False),
        StructField("page_text", StringType(), True),
        StructField("char_count", IntegerType(), False),
        StructField("extraction_method", StringType(), False),
    ]
)


def write_extracted_pages(catalog: str, schema: str, local_pdf_dir: str) -> int:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume="unused")
    extractor = TextExtractor()

    rows = []
    for pdf in sorted(Path(local_pdf_dir).glob("*.pdf")):
        text, pages = extractor.extract(str(pdf))
        print(f"Extracted {len(pages)} pages from {pdf.name} ({len(text)} chars)")
        for page_index, page_text in enumerate(pages, start=1):
            rows.append(
                Row(
                    document_id=pdf.stem,
                    file_name=pdf.name,
                    page_number=int(page_index),
                    page_text=page_text,
                    char_count=int(len(page_text)),
                    extraction_method="pymupdf-local",
                )
            )

    if not rows:
        raise FileNotFoundError(f"No PDFs found in {local_pdf_dir}")

    df = (
        spark.createDataFrame(rows, schema=EXTRACTED_PAGES_SCHEMA)
        .withColumn("page_number", col("page_number").cast("int"))
        .withColumn("char_count", col("char_count").cast("int"))
        .withColumn("extracted_at", current_timestamp())
    )

    table_name = target.table("silver_extracted_pages")
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
    parser.add_argument("--local-pdf-dir", required=True)
    args = parser.parse_args()
    write_extracted_pages(args.catalog, args.schema, args.local_pdf_dir)


if __name__ == "__main__":
    main()
