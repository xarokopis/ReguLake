import re
from pathlib import Path
from time import perf_counter

from gtgh_team3_compliance_assistant.processing.text_extractor import TextExtractor
from gtgh_team3_compliance_assistant.processing.eur_chunker import EurChunker
from gtgh_team3_compliance_assistant.processing.chunk_storing import ChunkStore
from gtgh_team3_compliance_assistant.processing.text_storing import ExtractedTextStore
from gtgh_team3_compliance_assistant.logger.Logger import log


LAW_PASSED_DATES = {
    "32016R0679_EN": "2016-04-27",
    "32014L0065_EN": "2014-05-15",
}


class IngestionService:
    def __init__(self):
        self.extractor = TextExtractor()
        self.chunker = EurChunker()
        self.chunk_store = ChunkStore()
        self.extracted_store = ExtractedTextStore()

    def process_local_pdf(self, pdf_path: str):
        start_time = perf_counter()
        pdf_path = Path(pdf_path)
        document_name = pdf_path.stem

        log.info(f"Processing document: {document_name}", file_path=str(pdf_path))

        with log.timer("pdf text extraction", document_name=document_name):
            text, pages = self.extractor.extract(str(pdf_path))

        log.info("Extracted text successfully", char_count=len(text), page_count=len(pages))

        extracted_path = self.extracted_store.save(document_name, text)
        log.debug(f"Saved extracted text raw dump", destination=str(extracted_path))

        with log.timer("document chunking", document_name=document_name):
            chunks = self.chunker.chunk(text)
        self._tag_pages(chunks, pages, document_name)

        with log.timer("saving chunks to store", document_name=document_name):
            self.chunk_store.save(
                document_id=document_name,
                chunks=chunks,
                source_pdf=str(pdf_path),
                law_passed_date=LAW_PASSED_DATES.get(document_name),
            )

        skipped_chunks = sum(1 for c in chunks if c.get("page") is None)
        duration = perf_counter() - start_time
        log.log_ingestion(
            filename=pdf_path.name,
            chunks=len(chunks),
            skipped=skipped_chunks,
            duration_s=duration
        )

        return {
            "document_name": document_name,
            "extracted_file": str(extracted_path),
            "chunk_count": len(chunks),
        }

    def _tag_pages(self, chunks, pages, document_name: str):
        normalized_pages = [re.sub(r"\s+", " ", p).lower() for p in pages]

        for idx, chunk in enumerate(chunks):
            page_num = self._find_page(chunk, normalized_pages)
            chunk["page"] = self._find_page(chunk, normalized_pages)

            if page_num is None:
                log.warning(
                    f"Orphan chunk detected: alignment failed", 
                    document=document_name,
                    chunk_index=idx,
                    chunk_type=chunk.get("type"),
                    article_number=chunk.get("article_number")
                )

    def _find_page(self, chunk, normalized_pages):
        text = chunk["text"]
        part_index = chunk.get("part_index", 0)

        if part_index > 0 and chunk.get("article_number"):
            match = re.search(r"(?:\(\w+\)|\d+\.)\s", text)
            if match:
                text = text[match.start():]
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

        normalized_text = re.sub(r"\s+", " ", text).lower().strip()

        if not normalized_text:
            return None

        for fp_len in (200, 100, 50):
            fingerprint = normalized_text[:fp_len]
            if not fingerprint:
                continue
            for page_num, normalized in enumerate(normalized_pages, start=1):
                if fingerprint in normalized:
                    return page_num

        return None
