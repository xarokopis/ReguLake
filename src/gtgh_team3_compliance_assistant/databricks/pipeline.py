from __future__ import annotations
import argparse
from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark
from gtgh_team3_compliance_assistant.databricks.setup_tables import setup_tables
from gtgh_team3_compliance_assistant.databricks.bronze import write_bronze
from gtgh_team3_compliance_assistant.databricks.silver_extract import write_extracted_pages
from gtgh_team3_compliance_assistant.databricks.silver_chunks import write_chunks
from gtgh_team3_compliance_assistant.databricks.gold import write_gold
from gtgh_team3_compliance_assistant.databricks.embeddings import write_embeddings


def run_pipeline(catalog: str, schema: str, volume: str, local_pdf_dir: str) -> None:
    target = DatabricksTarget(catalog=catalog, schema=schema, volume=volume)

    print("Setup tables and volume")
    setup_tables(catalog, schema, volume)

    print("Write Bronze document metadata")
    write_bronze(catalog, schema, volume, local_pdf_dir)

    print("Extract local PDF pages and write Silver pages")
    write_extracted_pages(catalog, schema, local_pdf_dir)

    print("Chunk extracted pages and write Silver chunks")
    write_chunks(catalog, schema)

    print("Create Gold AI-ready chunks")
    write_gold(catalog, schema)

    print("Create local embeddings and write Gold embedded chunks")
    write_embeddings(catalog, schema)

    spark = get_spark()
    print("\nFinal table counts:")
    for table in [
        "bronze_raw_documents",
        "silver_extracted_pages",
        "silver_chunks",
        "gold_ai_ready_chunks",
        "gold_embedded_chunks",
    ]:
        count = spark.table(target.table(table)).count()
        print(f" - {target.table(table)}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--local-pdf-dir", default="data/pdfs")
    args = parser.parse_args()
    run_pipeline(args.catalog, args.schema, args.volume, args.local_pdf_dir)


if __name__ == "__main__":
    main()