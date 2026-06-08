import json

from gtgh_team3_compliance_assistant.config import METADATA_FILE


class MetadataStore:
    def __init__(self):
        if not METADATA_FILE.exists():
            METADATA_FILE.write_text("[]")

        self.file = METADATA_FILE

    def load(self):
        return json.loads(self.file.read_text())

    def save(self, data):
        self.file.write_text(json.dumps(data, indent=2))

    def add(self, document: dict):
        data = self.load()
        data.append(document)
        self.save(data)

    def exists(self, url: str) -> bool:
        data = self.load()
        return any(d["url"] == url for d in data)
