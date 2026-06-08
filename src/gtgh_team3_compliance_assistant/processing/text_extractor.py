import re
import fitz
from datetime import datetime


class TextExtractor:
    def extract_metadata(self, first_page_text: str) -> dict:
        text = re.sub(r'\s+', ' ', first_page_text).strip()

        version_match = re.search(r'\(EU\)\s+([\d]{4}/[\d]+(?:/EU)?)', text)
        if not version_match:
            version_match = re.search(r'(?:REGULATION|DIRECTIVE)\s+([\d]{4}/[\d]+/EU)', text)
        document_version = version_match.group(1) if version_match else None

        date_match = re.search(r'\bof\s+(\d{1,2}\s+\w+\s+\d{4})\b', text)
        law_passed_date = None
        if date_match:
            try:
                law_passed_date = datetime.strptime(date_match.group(1), '%d %B %Y').strftime('%Y-%m-%d')
            except ValueError:
                pass

        issuing_authority = None
        if 'EUROPEAN PARLIAMENT AND OF THE COUNCIL' in text:
            issuing_authority = 'European Parliament and Council of the European Union'

        title_match = re.search(r'\(([A-Z][A-Za-z\s\-]+(?:Act|Regulation|Directive|Framework|Mechanism))\)', text)
        if title_match:
            regulation_title = title_match.group(1).strip()
        else:
            type_match = re.search(r'^(REGULATION|DIRECTIVE)', text)
            regulation_title = f'{type_match.group(1).capitalize()} {document_version}' if type_match and document_version else None

        return {
            'regulation_title': regulation_title,
            'document_version': document_version,
            'law_passed_date': law_passed_date,
            'issuing_authority': issuing_authority,
        }

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
