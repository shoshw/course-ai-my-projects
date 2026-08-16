"""Manages the corpus of uploaded documents."""

import os

from langchain_cohere import CohereEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from core.sources import chunk_source

_EMBEDDING_MODEL = os.getenv("NOTEBOOKLM_EMBEDDING_MODEL", "embed-multilingual-v3.0")


class SourceStore:
    def __init__(self) -> None:
        self.sources = []
        self.vector_store = InMemoryVectorStore(CohereEmbeddings(model=_EMBEDDING_MODEL))

    def add(self, name: str, content: str) -> None:
        self.sources.append(content)
        chunks = chunk_source(content)
        self.vector_store.add_texts(chunks)
