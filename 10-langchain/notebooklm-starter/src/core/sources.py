"""Functions for handling source documents and texts."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


def chunk_source(name: str, content: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    return splitter.create_documents([content], metadatas=[{"source": name}])


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )
