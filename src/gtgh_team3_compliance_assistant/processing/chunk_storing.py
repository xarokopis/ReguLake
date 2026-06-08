import json
from datetime import datetime, timezone

from gtgh_team3_compliance_assistant.config import CHUNK_DIR


class ChunkStore:
    def save(
        self,
        document_id: str,
        chunks: list[dict],
        source_pdf: str | None = None,
        law_passed_date: str | None = None,
        doc_meta: dict | None = None,
    ):
        file_path = CHUNK_DIR / f"{document_id}.json"
        meta = doc_meta or {}

        payload = {
            "document_id": document_id,
            "source_pdf": source_pdf,
            "regulation_title": meta.get("regulation_title"),
            "document_version": meta.get("document_version"),
            "issuing_authority": meta.get("issuing_authority"),
            "law_passed_date": meta.get("law_passed_date") or law_passed_date,
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
            "chunks": [],
        }

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk["text"]

            payload["chunks"].append(
                {
                    "chunk_uid": self._build_uid(document_id, chunk, idx),
                    "chunk_id": idx,
                    "type": chunk.get("type"),
                    "article_number": chunk.get("article_number"),
                    "annex_number": chunk.get("annex_number"),
                    "title": chunk.get("title"),
                    "part_index": chunk.get("part_index", 0),
                    "part_count": chunk.get("part_count", 1),
                    "page": chunk.get("page"),
                    "text": chunk_text,
                    "char_length": len(chunk_text),
                }
            )

        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return file_path

    def _build_uid(self, document_id: str, chunk: dict, idx: int) -> str:
        chunk_type = chunk.get("type")
        part_index = chunk.get("part_index", 0)

        if chunk_type == "article" and chunk.get("article_number"):
            return f"{document_id}_art_{chunk['article_number']}_p_{part_index}"

        if chunk_type == "annex" and chunk.get("annex_number"):
            return f"{document_id}_ann_{chunk['annex_number']}_p_{part_index}"

        return f"{document_id}_chunk_{idx}"
