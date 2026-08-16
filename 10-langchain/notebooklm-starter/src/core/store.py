"""Manages the corpus of uploaded documents."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from core.sources import chunk_source

_EMBEDDING_MODEL = os.getenv("NOTEBOOKLM_EMBEDDING_MODEL", "embed-multilingual-v3.0")


@dataclass
class Source:
    id: str
    name: str
    content: str
    active: bool = True
    chunk_ids: list[str] = field(default_factory=list)


class SourceStore:
    def __init__(self) -> None:
        self.sources: list[Source] = []
        self.vector_store = InMemoryVectorStore(CohereEmbeddings(model=_EMBEDDING_MODEL))

    def add(self, name: str, content: str) -> Source:
        chunks = chunk_source(name, content)
        chunk_ids = self.vector_store.add_documents(chunks)
        source = Source(id=uuid.uuid4().hex[:8], name=name, content=content, chunk_ids=chunk_ids)
        self.sources.append(source)
        return source

    def remove(self, source_id: str) -> bool:
        source = self.get(source_id)
        if source is None:
            return False
        self.vector_store.delete(ids=source.chunk_ids)
        self.sources.remove(source)
        return True

    def search(self, query: str, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def list(self) -> list[Source]:
        return self.sources

    def get(self, source_id: str) -> Source | None:
        return next((s for s in self.sources if s.id == source_id), None)

    def set_active(self, source_id: str, active: bool) -> Source | None:
        source = self.get(source_id)
        if source is None:
            return None
        source.active = active
        return source

    def active_ids(self) -> list[str]:
        return [s.id for s in self.sources if s.active]


store = SourceStore()
