from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DatabricksTarget:
    catalog: str
    schema: str
    volume: str

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def pdf_volume_dir(self) -> str:
        return f"{self.volume_root}/raw/pdfs"

    def table(self, name: str) -> str:
        return f"{self.namespace}.{name}"