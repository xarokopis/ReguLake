import re
from datetime import datetime, timezone
from pathlib import Path

from gtgh_team3_compliance_assistant.processing.text_extractor import TextExtractor
from gtgh_team3_compliance_assistant.processing.eur_chunker import EurChunker
from gtgh_team3_compliance_assistant.processing.chunk_storing import ChunkStore
from gtgh_team3_compliance_assistant.ingestion.metadata_storing import MetadataStore
from gtgh_team3_compliance_assistant.models.chunks import ChunkInput, AddChunksInput
from gtgh_team3_compliance_assistant.models.search import SearchInput
from gtgh_team3_compliance_assistant.logger.Logger import log

class RAGPipeline:

    def __init__(self, pdf_path, embedding_model, vector_store, llm=None):
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.extractor = TextExtractor()
        self.chunker = EurChunker()
        self.chunk_store = ChunkStore()
        log.info("RAGPipeline initialized", pdf=str(self.pdf_path), llm=llm is not None)

    def ingest(self):
        if self.pdf_path is None:
            log.error("ingest() called but pdf_path is not set")
            raise ValueError("pdf_path is not set")

        log.info("Ingestion started", file=self.pdf_path.name)
        # Extract Text from pdf
        doc_meta, ingested_at, raw_chunks = self.chunk_pdf()

        # Save Chunk Data to Json file
        self.chunk_store.save(
            document_id=self.pdf_path.stem,
            chunks=raw_chunks,
            source_pdf=str(self.pdf_path),
            doc_meta=doc_meta,
        )
        log.info("Chunks saved to disk", file=self.pdf_path.stem)

        chunk_models = self._create_chunk_models(raw_chunks, doc_meta, ingested_at)
        log.info("Chunk models created", count=len(chunk_models))
        
        with log.timer("embedding", file=self.pdf_path.name):
            embeddings = self.embedding_model.embed_documents(
                [chunk.text for chunk in chunk_models]
            )

        if len(embeddings) != len(chunk_models): 
            log.warning("Embedding/chunk count mismatch", chunks=len(chunk_models), embeddings=len(embeddings))

        add_input = AddChunksInput(chunks=chunk_models, embeddings=embeddings)
        self.vector_store.add_chunks(add_input)
        log.info("Chunks uploaded to vector store", count=len(chunk_models))

        MetadataStore().add({
            "document_id": self.pdf_path.stem,
            "source_pdf": str(self.pdf_path),
            "regulation_title": doc_meta.get("regulation_title"),
            "document_version": doc_meta.get("document_version"),
            "issuing_authority": doc_meta.get("issuing_authority"),
            "law_passed_date": doc_meta.get("law_passed_date"),
            "ingested_at": ingested_at,
            "chunk_count": len(chunk_models),
        })
        log.info("Metadata stored", document_id=self.pdf_path.stem)

    def chunk_pdf(self):
        log.info("Extracting text from PDF", file=self.pdf_path.name)
        text, pages = self.extractor.extract(str(self.pdf_path))
        log.info("Text extracted", chars=len(text), pages=len(pages))

        doc_meta = self.extractor.extract_metadata(pages[0])
        log.debug("Metadata extracted", regulation=doc_meta.get("regulation_title"), version=doc_meta.get("document_version"))
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        raw_chunks = self.chunker.chunk(text)
        log.info("PDF chunked", chunk_count=len(raw_chunks))

        self._tag_pages(raw_chunks, pages)
        return doc_meta, ingested_at, raw_chunks

    def retrieve(self, question: str, top_k: int = 5):
        log.info("Retrieving chunks", question=question[:120], top_k=top_k)
        with log.timer("vector search"):
            results = self.vector_store.search(
                SearchInput(query_embedding=self.embedding_model.embed_query(question), top_k=top_k)
            )
        log.info("Retrieval complete", hits=len(results))
        return results

    def build_context(self, results):
        if not results:
            log.warning("build_context called with empty results")

        parts = []
        for r in results:
            m = r.metadata
            title = m.get('regulation_title') or m.get('source_file', 'Unknown')
            version = m.get('document_version')
            doc_ref = f"{title} ({version})" if version else title

            article = m.get('article_number')
            annex = m.get('annex_number')
            page = m.get('page', m.get('page_number', '?'))
            part_index = m.get('part_index', 0)
            part_count = m.get('part_count', 1)

            if article:
                location = f"Article {article}"
                if part_count > 1:
                    location += f" — Part {part_index + 1}/{part_count}"
            elif annex:
                location = f"Annex {annex}"
            else:
                location = f"Page {page}"

            header = f"[{doc_ref} | {location} | Page {page}]"
            parts.append(f"{header}\n{r.content}")
        
        context = "\n\n---\n\n".join(parts)
        log.debug("Context built", chunks=len(results), chars=len(context))
        return context

    def ask(self, question: str, top_k: int = 5):
        if self.llm is None:
            log.error("ask() called but no LLM is configured")
            raise ValueError("No LLM configured")
        log.info("Ask started", question=question[:120])

        results = self.retrieve(question, top_k)
        context = self.build_context(results)
        with log.timer("llm generation"):
            answer = self.llm.generate(question=question, context=context)
        
        log.info("Answer generated", question=question[:120])

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": [r.model_dump() for r in results],
        }

    def _tag_pages(self, chunks, pages):
        normalized_pages = [re.sub(r"\s+", " ", p).lower() for p in pages]
        for chunk in chunks:
            chunk["page"] = self._find_page(chunk, normalized_pages)

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

    def _build_uid(self, document_id: str, chunk: dict, idx: int) -> str:
        chunk_type = chunk.get("type")
        part_index = chunk.get("part_index", 0)

        if chunk_type == "article" and chunk.get("article_number"):
            return f"{document_id}_art_{chunk['article_number']}_p_{part_index}"

        if chunk_type == "annex" and chunk.get("annex_number"):
            return f"{document_id}_ann_{chunk['annex_number']}_p_{part_index}"

        return f"{document_id}_chunk_{idx}"

    def _create_chunk_models(self, raw_chunks: list, metadata: dict, ingested_at: str) -> list:
        returned_list = []
        failed = 0
        for idx, chunk in enumerate(raw_chunks):
            chunk_uid = self._build_uid(self.pdf_path.stem, chunk, idx)

            try:
                model = ChunkInput(
                    chunk_uid=chunk_uid,
                    chunk_id=idx,
                    type=chunk.get("type"),
                    article_number=chunk.get("article_number"),
                    annex_number=chunk.get("annex_number"),
                    title=chunk.get("title"),
                    part_index=chunk.get("part_index", 0),
                    part_count=chunk.get("part_count", 1),
                    text=chunk["text"],
                    source_file=self.pdf_path.name,
                    page=chunk.get("page"),
                    char_length=len(chunk["text"]),
                    law_passed_date=metadata.get("law_passed_date"),
                    ingested_at=ingested_at,
                    regulation_title=metadata.get("regulation_title"),
                    document_version=metadata.get("document_version"),
                    issuing_authority=metadata.get("issuing_authority"),
                )
                returned_list.append(model)

            except Exception as e:
                failed += 1
                log.error("Chunk model creation failed", exc=e, chunk_id=idx, chunk=chunk)
                raise e
        
        log.info("Chunk models built", total=len(returned_list), failed=failed)
        return returned_list