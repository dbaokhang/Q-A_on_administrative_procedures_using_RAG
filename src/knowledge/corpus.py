"""
Dựng tập chunk từ registry (bước 2–5 của Bảng 2.10): trích xuất → phân đoạn theo cấu trúc → gắn metadata.

Chỉ tài liệu ở trạng thái approved/indexed được đưa vào (BR-01). Mỗi chunk mang đủ metadata để lọc quyền,
lọc hiệu lực và tạo trích dẫn (document_id, version_id, chunk_id, access_scope, effective_from/to...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

try:
    from ..chunker import split_documents
    from ..document_upload import extract_text
    from .registry import DocumentRegistry
    from .schema import DocumentRecord, date_ordinal, stable_chunk_id
    from .tthc import load_procedures, procedure_chunks
except ImportError:
    from chunker import split_documents
    from document_upload import extract_text
    from knowledge.registry import DocumentRegistry
    from knowledge.schema import DocumentRecord, date_ordinal, stable_chunk_id
    from knowledge.tthc import load_procedures, procedure_chunks

# Tăng khi đổi cách phân đoạn / metadata → phiên bản kho đổi → chỉ mục được dựng lại
CHUNKER_VERSION = "2026.11-1"


def _text_chunks(record: DocumentRecord, path: Path) -> list[Document]:
    text = extract_text(path.name, path.read_bytes())
    base = Document(page_content=text, metadata={"source": path.name, "label": record.title})
    chunks = split_documents([base])
    shared = record.chunk_metadata()
    for index, chunk in enumerate(chunks):
        article = chunk.metadata.get("article") or ""
        if not article:
            # Tài liệu không chia theo "Điều N" (hướng dẫn nội bộ): dùng dòng tiêu đề đầu đoạn làm vị trí
            lines = chunk.page_content.split("\n", 2)
            article = lines[1].strip()[:80] if len(lines) > 1 else ""
        superseded_from = chunk.metadata.get("superseded_from")
        chunk.metadata = {
            **shared,
            "source": path.name,
            "chunk_id": stable_chunk_id(record.version_id, index),
            "chunk_index": index,
            "article": article,
            "article_reference": article,
            "superseded": bool(chunk.metadata.get("superseded")),
            "superseded_from": superseded_from,
            "superseded_from_ord": date_ordinal(superseded_from),
        }
    return chunks


def chunks_for_record(record: DocumentRecord, registry: DocumentRegistry) -> list[Document]:
    path = registry.resolve_path(record)
    if record.document_type == "administrative_procedure":
        chunks = procedure_chunks(record, load_procedures(path))
        for chunk in chunks:
            chunk.metadata.update(source=path.name, superseded=False, superseded_from=None, superseded_from_ord=None)
        return chunks
    return _text_chunks(record, path)


class MissingSourceError(FileNotFoundError):
    """Tệp nguồn của tài liệu đang phục vụ không còn trên đĩa và chỉ mục hiện tại cũng không giữ bản nào của nó."""


def missing_sources(registry: DocumentRegistry, records: list[DocumentRecord]) -> list[DocumentRecord]:
    return [r for r in records if not registry.resolve_path(r).is_file()]


def build_chunks(registry: DocumentRegistry, records: list[DocumentRecord] | None = None,
                 indexed: Callable[[], list[Document]] | None = None,
                 log: Callable[[str], None] = print) -> list[Document]:
    """Trích xuất + phân đoạn mọi tài liệu phục vụ.

    indexed: hàm đọc các chunk đang có trong chỉ mục. Khi tệp nguồn bị thiếu nhưng chỉ mục còn đúng phiên bản đó
    (version_id = mã tài liệu @ hash nội dung tệp), chunk cũ được dùng lại (metadata làm mới theo registry) thay vì
    dừng hoặc bỏ tài liệu — lần dựng lại sẽ xóa collection cũ, nên đó có thể là bản duy nhất còn lại của nội dung.
    Thiếu tệp mà chỉ mục cũng không có → MissingSourceError (không âm thầm phục vụ kho thiếu văn bản).
    """
    records = records if records is not None else registry.servable()
    missing = {r.document_id for r in missing_sources(registry, records)}
    reusable: dict[str, list[Document]] = {}
    if missing and indexed is not None:
        wanted = {r.version_id for r in records if r.document_id in missing}
        for chunk in indexed():
            version_id = chunk.metadata.get("version_id")
            if version_id in wanted:
                reusable.setdefault(version_id, []).append(chunk)
    lost = [r for r in records if r.document_id in missing and r.version_id not in reusable]
    if lost:
        listing = "\n".join(f"  - {r.file_path}  ({r.document_id})" for r in lost)
        raise MissingSourceError(
            f"Thiếu {len(lost)} tệp nguồn của tài liệu đang phục vụ và chỉ mục không còn bản nào để dùng lại:\n"
            f"{listing}\nChép lại đúng các tệp này vào vị trí trên, hoặc chuyển tài liệu sang trạng thái khác "
            "trong trang quản trị trước khi đánh chỉ mục lại."
        )

    chunks: list[Document] = []
    for record in records:
        if record.document_id not in missing:
            chunks.extend(chunks_for_record(record, registry))
            continue
        kept = sorted(reusable[record.version_id], key=lambda c: c.metadata.get("chunk_index", 0))
        shared = record.chunk_metadata()
        for chunk in kept:
            chunk.metadata.update(shared)
        log(f"Cảnh báo: thiếu tệp {record.file_path} — dùng lại {len(kept)} đoạn đã đánh chỉ mục "
            f"({record.version_id}). Hãy chép lại tệp gốc vào data/.")
        chunks.extend(kept)
    return chunks


def corpus_version(records: list[DocumentRecord]) -> str:
    """Định danh phiên bản kho phục vụ: đổi khi tài liệu, metadata dùng để lọc/trích dẫn hoặc chunker đổi."""
    keys = ("version_id", "title", "document_number", "effective_from", "effective_to", "legal_status",
            "access_scope", "source_url", "field_area")
    payload = [{k: getattr(r, k) for k in keys} for r in sorted(records, key=lambda r: r.document_id)]
    digest = hashlib.sha256(json.dumps([CHUNKER_VERSION, payload], ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()[:16]
