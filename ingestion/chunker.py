from pathlib import Path
from typing import List
import pdfplumber                          # pip install pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single chunk of text with its provenance metadata."""
    content: str
    source: str                            # original file path
    page: int | None = None               # PDF page number, if applicable
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Loads raw text from .pdf and .txt files."""

    def load(self, path: str | Path) -> tuple[str, dict]:
        """Returns (full_text, base_metadata) for a given file."""
        path = Path(path)

        if path.suffix == ".pdf":
            return self._load_pdf(path)
        elif path.suffix == ".txt":
            return self._load_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_pdf(self, path: Path) -> tuple[str, dict]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        full_text = "\n\n".join(pages)
        return full_text, {"source": str(path), "total_pages": len(pages)}

    def _load_txt(self, path: Path) -> tuple[str, dict]:
        full_text = path.read_text(encoding="utf-8")
        return full_text, {"source": str(path)}


class Chunker:
    """
    Splits documents into overlapping chunks using recursive character splitting.

    chunk_size:    target character count per chunk (~500 ≈ 100 tokens)
    chunk_overlap: shared characters between adjacent chunks to preserve context
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        )
        self.loader = DocumentLoader()

    def chunk_file(self, path: str | Path) -> List[Document]:
        """Load a file and return its chunks as Document objects."""
        path = Path(path)
        full_text, base_metadata = self.loader.load(path)

        raw_chunks = self.splitter.split_text(full_text)

        return [
            Document(
                content=chunk,
                source=str(path),
                chunk_index=i,
                metadata={**base_metadata, "chunk_index": i},
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    def chunk_directory(self, dir_path: str | Path) -> List[Document]:
        """Recursively chunk all .pdf and .txt files in a directory."""
        dir_path = Path(dir_path)
        all_docs = []

        for path in sorted(dir_path.rglob("*")):
            if path.suffix in {".pdf", ".txt"}:
                print(f"  Loading: {path.name} ...", end=" ")
                docs = self.chunk_file(path)
                print(f"{len(docs)} chunks")
                all_docs.extend(docs)

        return all_docs
