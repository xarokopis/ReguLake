from __future__ import annotations
import argparse
from pathlib import Path
from pyspark.sql import Row
from pyspark.sql.functions import current_timestamp
from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark


REGULATION_FAMILIES = {
    "32014L0065_EN": "MiFID II",
    "32015L2366_EN": "PSD2",
    "32016R0679_EN": "GDPR",
    "32024L1640_EN": "AMLD",
    "32024R1689_EN": "AI Act",
}


def write_bronze(catalog: str, schema: str, volume: str, local_pdf_dir: str) -> int:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume=volume)

    pdfs = sorted(Path(local_pdf_dir).glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {local_pdf_dir}")

    rows = []

    for pdf in pdfs:
        document_id = pdf.stem

        rows.append(
            Row(
                document_id=document_id,
                file_name=pdf.name,
                source_path=f"{target.pdf_volume_dir}/{pdf.name}",
                source_name="EUR-Lex",
                regulation_family=REGULATION_FAMILIES.get(document_id),
                file_type="pdf",
                status="new",
            )
        )

    df = spark.createDataFrame(rows).withColumn(
        "ingestion_timestamp",
        current_timestamp(),
    )

    table_name = target.table("bronze_raw_documents")

    df.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(table_name)

    count = df.count()
    print(f"Wrote {count} rows to {table_name}")

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--local-pdf-dir", required=True)

    args = parser.parse_args()

    write_bronze(
        catalog=args.catalog,
        schema=args.schema,
        volume=args.volume,
        local_pdf_dir=args.local_pdf_dir,
    )


if __name__ == "__main__":
    main()