import fitz


class TextExtractor:
    def extract(self, file_path: str):
        doc = fitz.open(file_path)

        pages = []

        try:
            for page in doc:
                pages.append(page.get_text("text"))
        finally:
            doc.close()

        full_text = "\n".join(pages)

        return full_text, pages
