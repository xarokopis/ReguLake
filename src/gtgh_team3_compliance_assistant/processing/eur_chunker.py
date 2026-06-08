import re


class EurChunker:
    ARTICLE_PATTERN = r"(?m)^\s*Article\s+(\d+[A-Za-z]?)\s*$"
    ANNEX_PATTERN = r"(?m)^\s*ANNEX(?:ES)?\b"
    ANNEX_HEADING_PATTERN = r"(?m)^\s*ANNEX(?:ES)?\s+([IVX]+)?\s*$"
    PARAGRAPH_SPLIT_PATTERN = r"\n(?=(?:\([A-Za-z0-9]+\)|\d+\.)\s)"

    def __init__(self, max_chunk_chars: int = 4000, overlap: int = 50):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

    def chunk(self, text: str):
        text = self._clean_text(text)

        if not text:
            return []

        body_text, annex_text = self._split_body_and_annexes(text)

        article_chunks = self._chunk_articles(body_text)
        annex_chunks = self._chunk_annexes(annex_text) if annex_text else []

        if article_chunks:
            return article_chunks + annex_chunks

        return self._fallback_chunk(body_text) + annex_chunks

    def _clean_text(self, text: str):
        text = text.replace("­", "")
        text = text.replace("￾", "")
        text = text.replace("￾", "")

        text = re.sub(r"(?m)^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$", "", text)
        text = re.sub(r"(?m)^\s*L \d+/\d+\s*$", "", text)
        text = re.sub(r"(?m)^\s*EN\s*$", "", text)
        text = re.sub(r"(?m)^\s*Official Journal of the European Union\s*$", "", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize(self, text: str):
        return re.sub(r"\s+", " ", text).strip()

    def _split_body_and_annexes(self, text: str):
        match = re.search(self.ANNEX_PATTERN, text)
        if not match:
            return text, ""
        return text[: match.start()].rstrip(), text[match.start():].strip()

    def _chunk_articles(self, text: str):
        matches = list(re.finditer(self.ARTICLE_PATTERN, text, flags=re.IGNORECASE))

        if not matches:
            return []

        chunks = []

        for i, match in enumerate(matches):
            original_start = match.start()
            original_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            number = match.group(1)
            article_body = text[original_start:original_end].strip()
            title = self._get_article_title(article_body)

            if len(article_body) <= self.max_chunk_chars:
                if i == 0:
                    ctx_start = original_start
                else:
                    ctx_start = max(0, original_start - self.overlap)

                parts = [text[ctx_start:original_end].strip()]
            else:
                parts = self._split_oversized_article(article_body, number, title)

            for idx, part_text in enumerate(parts):
                chunks.append(
                    {
                        "type": "article",
                        "article": f"Article {number}",
                        "article_number": number,
                        "recital_number": None,
                        "annex_number": None,
                        "title": title,
                        "part_index": idx,
                        "part_count": len(parts),
                        "text": self._normalize(part_text),
                    }
                )

        return chunks

    def _split_oversized_article(self, article_body: str, number: str, title: str | None):
        paragraphs = re.split(self.PARAGRAPH_SPLIT_PATTERN, article_body)

        if len(paragraphs) <= 1:
            return self._hard_split(article_body)

        header = f"Article {number}" + (f" {title}" if title else "")

        grouped = []
        current = f"{paragraphs[0]}\n{paragraphs[1]}"

        for paragraph in paragraphs[2:]:
            candidate_len = len(current) + len(paragraph) + 1

            if candidate_len <= self.max_chunk_chars:
                current = f"{current}\n{paragraph}"
            else:
                grouped.append(current)
                current = f"{header}\n{paragraph}"

        if current.strip():
            grouped.append(current)

        out = []
        for chunk in grouped:
            out.extend(self._hard_split(chunk))
        return out

    def _hard_split(self, text: str):
        if len(text) <= self.max_chunk_chars:
            return [text]

        parts = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_chars
            parts.append(text[start:end])
            start = end - self.overlap

        return parts

    def _get_article_title(self, article_body: str):
        lines = [line.strip() for line in article_body.splitlines() if line.strip()]

        if len(lines) < 2:
            return None

        for line in lines[1:4]:
            if self._is_likely_title(line):
                return line

        return None

    def _is_likely_title(self, line: str):
        if not line or len(line) > 150:
            return False
        if re.match(r"^\d+\.", line):
            return False
        if re.match(r"^\(\d+\)", line):
            return False
        if re.match(r"^\d{1,2}\.\d{1,2}\.\d{4}", line):
            return False
        if line.startswith("L ") and re.match(r"^L \d+/\d+", line):
            return False
        if "Official Journal" in line:
            return False
        if line == "EN":
            return False
        return True

    def _chunk_annexes(self, text: str):
        matches = list(re.finditer(self.ANNEX_HEADING_PATTERN, text))

        if not matches:
            return self._wrap_annex_chunks(text, annex_number=None)

        chunks = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            annex_number = match.group(1)
            annex_text = text[start:end].strip()

            chunks.extend(self._wrap_annex_chunks(annex_text, annex_number))

        return chunks

    def _wrap_annex_chunks(self, annex_text: str, annex_number: str | None):
        parts = self._hard_split(annex_text)
        title = f"Annex {annex_number}" if annex_number else "Annex"

        return [
            {
                "type": "annex",
                "article": None,
                "article_number": None,
                "recital_number": None,
                "annex_number": annex_number,
                "title": title,
                "part_index": idx,
                "part_count": len(parts),
                "text": self._normalize(part_text),
            }
            for idx, part_text in enumerate(parts)
        ]

    def _fallback_chunk(self, text: str, chunk_size: int = 1000, overlap: int = 150):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = self._normalize(text[start:end])

            if chunk_text:
                chunks.append(
                    {
                        "type": "fallback",
                        "article": None,
                        "article_number": None,
                        "recital_number": None,
                        "annex_number": None,
                        "title": None,
                        "part_index": 0,
                        "part_count": 1,
                        "text": chunk_text,
                    }
                )

            start = end - overlap

        return chunks
