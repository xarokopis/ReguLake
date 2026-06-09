from __future__ import annotations

import argparse

from pyspark.sql.functions import col, current_timestamp, lit

from gtgh_team3_compliance_assistant.config import EMBEDDING_MODEL_NAME
from gtgh_team3_compliance_assistant.databricks.paths import DatabricksTarget
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder


NUMERIC_COLUMNS = [
    "chunk_id",
    "page_number",
    "part_index",
    "part_count",
    "char_length",
]


def write_embeddings(catalog: str, schema: str, batch_size: int = 64) -> int:
    spark = get_spark()
    target = DatabricksTarget(catalog=catalog, schema=schema, volume="unused")

    rows = spark.table(target.table("gold_ai_ready_chunks")).orderBy("chunk_uid").collect()
    if not rows:
        raise ValueError("gold_ai_ready_chunks is empty")

    embedder = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
    output = []

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [row["chunk_text"] for row in batch]
        embeddings = embedder.embed_documents(texts)

        for row, embedding in zip(batch, embeddings):
            data = row.asDict()
            for column in NUMERIC_COLUMNS:
                if data.get(column) is not None:
                    data[column] = int(data[column])
            data["embedding"] = [float(value) for value in embedding]
            output.append(data)

        print(f"Embedded {min(start + batch_size, len(rows))}/{len(rows)} chunks")

    df = spark.createDataFrame(output)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast("int"))

    df = (
        df.withColumn("embedding_model", lit(EMBEDDING_MODEL_NAME))
        .withColumn("embedding_version", lit("local_sentence_transformers_v1"))
        .withColumn("embedded_at", current_timestamp())
    )

    table_name = target.table("gold_embedded_chunks")
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
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    write_embeddings(args.catalog, args.schema, args.batch_size)


if __name__ == "__main__":
    main()
