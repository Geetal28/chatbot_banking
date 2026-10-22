"""Load documents, split them into chunks and index them.

Run:  python -m app.ingest [--reset]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.retrieval import HybridRetriever, SentenceTransformerEmbedder

DOC_TYPE_PREFIXES = {
    "fees_": "fees",
    "terms_": "terms",
    "regulatory_": "regulatory",
    "faq_": "faq",
    "product_": "product",
}
SKIP_FILES = {"readme.md"}
SUPPORTED = {".pdf", ".txt", ".md"}


def infer_doc_type(filename: str) -> str:
    low = filename.lower()
    for prefix, kind in DOC_TYPE_PREFIXES.items():
        if low.startswith(prefix):
            return kind
    return "general"


def _pdf_pages(path: Path, extract_tables: bool = True) -> list[tuple[int, str]]:
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # older name

    pages: list[tuple[int, str]] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if extract_tables:
                try:
                    tables = page.find_tables()
                    md = [t.to_markdown() for t in tables.tables if t.row_count > 1]
                    if md:
                        text += "\n\nTables on this page:\n" + "\n\n".join(md)
                except Exception:
                    pass  # table extraction is best-effort; plain text is already captured
            if text.strip():
                pages.append((i, text))
            else:
                print(f"  warning: {path.name} page {i} has no text layer (scanned?). Skipped; OCR needed.")
    return pages


def load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED or path.name.lower() in SKIP_FILES:
            continue
        doc_type = infer_doc_type(path.name)
        if path.suffix.lower() == ".pdf":
            pages = _pdf_pages(path)
        else:
            pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
        for page_no, text in pages:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_no, "doc_type": doc_type}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Paragraph, line, sentence, word. The original project had '' before ' ',
        # which stopped the splitter from ever breaking on words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[tuple[str, int], int] = {}
    for c in chunks:
        key = (c.metadata["source"], c.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        c.metadata["chunk"] = counters[key]
    return chunks


def chunk_id(chunk: Document) -> str:
    m = chunk.metadata
    raw = f'{m["source"]}|{m["page"]}|{m["chunk"]}|{chunk.page_content}'
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def build_index(settings: Settings, retriever: HybridRetriever, reset: bool = False) -> dict[str, Any]:
    if reset:
        retriever.reset()
    docs = load_documents(settings.docs_dir)
    if not docs:
        return {"documents": 0, "chunks": 0, "message": f"No documents found in {settings.docs_dir}"}
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    ids = [chunk_id(c) for c in chunks]
    metas: list[dict[str, Any]] = []
    for c in chunks:
        meta = dict(c.metadata)
        meta["chars"] = len(c.page_content)
        metas.append(meta)
    retriever.add_chunks(ids, [c.page_content for c in chunks], metas)
    sources = sorted({d.metadata["source"] for d in docs})
    return {"documents": len(sources), "pages": len(docs), "chunks": len(chunks), "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents for the banking assistant")
    parser.add_argument("--reset", action="store_true", help="drop the existing index first")
    args = parser.parse_args()
    s = get_settings()
    retriever = HybridRetriever(SentenceTransformerEmbedder(s.embedding_model), s.index_dir)
    stats = build_index(s, retriever, reset=args.reset)
    print(stats)


if __name__ == "__main__":
    main()
