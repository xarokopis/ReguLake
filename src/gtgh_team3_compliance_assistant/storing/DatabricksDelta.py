from __future__ import annotations
import math
from pydantic import BaseModel, Field
from gtgh_team3_compliance_assistant.databricks.spark_session import get_spark


class DatabricksDeltaVectorStore(BaseModel):
    catalog: str
    db_schema: str = Field(alias="schema")
    table_name: str = "gold_embedded_chunks"

    class Config:
        populate_by_name = True

    def full_table_name(self) -> str:
        return f"{self.catalog}.{self.db_schema}.{self.table_name}"

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        spark = get_spark()
        table = self.full_table_name()

        rows = (
            spark.table(table)
            .select(
                "chunk_uid",
                "document_id",
                "file_name",
                "chunk_id",
                "page_number",
                "article_number",
                "title",
                "chunk_text",
                "char_length",
                "regulation_family",
                "source_name",
                "source_path",
                "embedding",
            )
            .collect()
        )

        def row_value(row, name, default=None):
            try:
                return row[name]
            except Exception:
                return default

        def normalize_embedding(value):
            if value is None:
                return []

            result = []

            for item in value:
                if item is None:
                    continue

                if isinstance(item, tuple):
                    if len(item) == 0:
                        continue
                    item = item[-1]

                if isinstance(item, list):
                    if len(item) == 0:
                        continue
                    item = item[-1]

                result.append(float(item))

            return result


        def cosine_similarity(a, b):
            a = normalize_embedding(a)
            b = normalize_embedding(b)

            if not a or not b:
                return 0.0

            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot / (norm_a * norm_b)

        scored = []

        for row in rows:
            embedding = row_value(row, "embedding", [])
            score = cosine_similarity(query_embedding, embedding)
            distance = 1.0 - score
            chunk_text = row_value(row, "chunk_text", "")

            scored.append(
                {
                    "chunk_uid": row_value(row, "chunk_uid"),
                    "document_id": row_value(row, "document_id"),
                    "file_name": row_value(row, "file_name"),
                    "source_file": row_value(row, "file_name"),
                    "chunk_id": row_value(row, "chunk_id"),
                    "page_number": row_value(row, "page_number"),
                    "page": row_value(row, "page_number"),
                    "article_number": row_value(row, "article_number"),
                    "title": row_value(row, "title"),
                    "chunk_text": chunk_text,
                    "text": chunk_text,
                    "char_length": row_value(row, "char_length"),
                    "regulation_family": row_value(row, "regulation_family"),
                    "source_name": row_value(row, "source_name"),
                    "source_path": row_value(row, "source_path"),
                    "score": score,
                    "distance": distance,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]