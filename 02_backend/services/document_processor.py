import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from pypdf import PdfReader


class DocumentProcessor:
    def __init__(self, data_dir: Path, chunk_size: int = 800, chunk_overlap: int = 120):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_path = self.data_dir / "documents.json"
        self.supported_extensions = {".pdf", ".txt", ".md"}
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def process_upload(self, file: BinaryIO, filename: str, content_type: str | None, size: int | None) -> tuple[dict, list[str]]:
        extension = Path(filename).suffix.lower()
        if extension not in self.supported_extensions:
            supported = ", ".join(sorted(self.supported_extensions))
            raise ValueError(f"Unsupported file type '{extension}'. Supported types: {supported}")

        document_id = str(uuid4())
        safe_filename = Path(filename).name
        stored_filename = f"{document_id}{extension}"
        stored_path = self.data_dir / stored_filename

        with stored_path.open("wb") as target:
            shutil.copyfileobj(file, target)

        text = self.extract_text(stored_path)
        chunks = self.chunk_text(text)
        if not chunks:
            stored_path.unlink(missing_ok=True)
            raise ValueError("No readable text could be extracted from the uploaded document")

        metadata = {
            "documentId": document_id,
            "filename": safe_filename,
            "storedPath": str(stored_path),
            "contentType": content_type,
            "size": size if size is not None else stored_path.stat().st_size,
            "chunkCount": len(chunks),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        documents = self.load_metadata()
        documents[document_id] = metadata
        self.save_metadata(documents)

        return metadata, chunks

    def list_documents(self) -> list[dict]:
        return sorted(
            self.load_metadata().values(),
            key=lambda document: document.get("createdAt", ""),
            reverse=True,
        )

    def delete_document(self, document_id: str) -> dict | None:
        documents = self.load_metadata()
        metadata = documents.pop(document_id, None)
        if metadata is None:
            return None

        stored_path = Path(metadata["storedPath"])
        stored_path.unlink(missing_ok=True)
        self.save_metadata(documents)
        return metadata

    def extract_text(self, file_path: Path) -> str:
        extension = file_path.suffix.lower()
        if extension in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8")
        if extension == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        raise ValueError(f"Unsupported file type '{extension}'")

    def chunk_text(self, text: str) -> list[str]:
        normalized = "\n".join(line.strip() for line in text.splitlines())
        normalized = "\n".join(line for line in normalized.splitlines() if line)
        if not normalized:
            return []

        chunks = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_length:
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def load_metadata(self) -> dict[str, dict]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def save_metadata(self, documents: dict[str, dict]) -> None:
        self.metadata_path.write_text(
            json.dumps(documents, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
