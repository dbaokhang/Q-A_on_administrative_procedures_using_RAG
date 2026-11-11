"""
Registry tài liệu: danh sách phiên bản tài liệu + nhật ký thao tác (FR-14, FR-15, BR-01, BR-08).

Lưu dạng JSON trong data/registry/ — đủ cho quy mô nguyên mẫu; ranh giới khái niệm Document / Version /
IngestionJob vẫn giữ trong các trường của DocumentRecord và audit log.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

try:
    from .schema import ALLOWED_TRANSITIONS, DocumentRecord, WORKFLOW_STATUSES, now_iso
except ImportError:
    from knowledge.schema import ALLOWED_TRANSITIONS, DocumentRecord, WORKFLOW_STATUSES, now_iso

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "data" / "registry"


class RegistryError(ValueError):
    """Thao tác không hợp lệ trên registry; thông điệp hiển thị được cho quản trị viên."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class DocumentRegistry:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or os.getenv("REGISTRY_DIR") or DEFAULT_REGISTRY_DIR)
        self.path = self.directory / "documents.json"
        self.audit_path = self.directory / "audit_log.jsonl"
        self._lock = threading.Lock()

    # ---- đọc / ghi ----
    def _load(self) -> list[DocumentRecord]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [DocumentRecord.from_dict(item) for item in json.load(f)]

    def _save(self, records: list[DocumentRecord]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    def _audit(self, action: str, actor: str, document_id: str, **details) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"at": now_iso(), "actor": actor, "action": action, "document_id": document_id, **details}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def resolve_path(self, record: DocumentRecord) -> Path:
        path = Path(record.file_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    # ---- truy vấn ----
    def all(self) -> list[DocumentRecord]:
        return self._load()

    def get(self, document_id: str) -> DocumentRecord:
        for record in self._load():
            if record.document_id == document_id:
                return record
        raise RegistryError(f"Không tìm thấy tài liệu {document_id!r}")

    def servable(self) -> list[DocumentRecord]:
        return [r for r in self._load() if r.servable]

    def audit_log(self, limit: int = 200) -> list[dict]:
        if not self.audit_path.exists():
            return []
        with open(self.audit_path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        return [json.loads(line) for line in reversed(lines) if line.strip()]

    # ---- thay đổi ----
    def register(self, record: DocumentRecord, actor: str) -> DocumentRecord:
        """Tiếp nhận tài liệu mới: tính hash, chặn tệp trùng nội dung, trạng thái ban đầu theo record."""
        with self._lock:
            records = self._load()
            if any(r.document_id == record.document_id for r in records):
                raise RegistryError(f"Mã tài liệu {record.document_id!r} đã tồn tại")
            if not record.file_hash:
                record.file_hash = file_sha256(self.resolve_path(record))
            record.version_id = record.version_id or f"{record.document_id}@{record.file_hash[:12]}"
            duplicate = next((r for r in records if r.file_hash == record.file_hash
                              and r.status not in {"withdrawn", "failed"}), None)
            if duplicate:
                raise RegistryError(f"Nội dung tệp trùng với tài liệu đã có: {duplicate.document_id}")
            records.append(record)
            self._save(records)
            self._audit("register", actor, record.document_id, status=record.status, version_id=record.version_id)
            return record

    def set_status(self, document_id: str, new_status: str, actor: str, note: str | None = None) -> DocumentRecord:
        if new_status not in WORKFLOW_STATUSES:
            raise RegistryError(f"Trạng thái không hợp lệ: {new_status!r}")
        with self._lock:
            records = self._load()
            record = next((r for r in records if r.document_id == document_id), None)
            if record is None:
                raise RegistryError(f"Không tìm thấy tài liệu {document_id!r}")
            if new_status not in ALLOWED_TRANSITIONS[record.status]:
                raise RegistryError(
                    f"Không thể chuyển {WORKFLOW_STATUSES[record.status]!r} → {WORKFLOW_STATUSES[new_status]!r}")
            old = record.status
            record.status = new_status
            record.updated_at = now_iso()
            if new_status == "approved":
                record.reviewed_by = actor
            self._save(records)
            self._audit("set_status", actor, document_id, old=old, new=new_status, note=note)
            return record

    def update_metadata(self, document_id: str, actor: str, **changes) -> DocumentRecord:
        """Sửa metadata (không sửa nội dung văn bản gốc, BR-08 ghi log). Metadata dùng để lọc / trích dẫn thay đổi
        thì phiên bản kho (corpus_version) đổi theo → lần đánh chỉ mục kế tiếp dựng lại các chunk liên quan."""
        protected = {"document_id", "file_hash", "version_id", "status", "ingested_at"}
        with self._lock:
            records = self._load()
            index = next((i for i, r in enumerate(records) if r.document_id == document_id), None)
            if index is None:
                raise RegistryError(f"Không tìm thấy tài liệu {document_id!r}")
            data = records[index].to_dict()
            applied = {k: v for k, v in changes.items() if k not in protected and data.get(k) != v}
            if not applied:
                return records[index]
            data.update(applied, updated_at=now_iso())
            records[index] = DocumentRecord.from_dict(data)  # kiểm tra lại giá trị hợp lệ
            self._save(records)
            self._audit("update_metadata", actor, document_id, changes=applied)
            return records[index]

    def supersede(self, old_id: str, new_id: str, actor: str) -> None:
        """Văn bản mới thay thế toàn bộ văn bản cũ: bản cũ rời truy vấn hiện hành nhưng vẫn giữ lịch sử."""
        with self._lock:
            records = self._load()
            old = next((r for r in records if r.document_id == old_id), None)
            new = next((r for r in records if r.document_id == new_id), None)
            if old is None or new is None:
                raise RegistryError("Không tìm thấy tài liệu cần liên kết thay thế")
            if "superseded" not in ALLOWED_TRANSITIONS[old.status]:
                raise RegistryError("Chỉ tài liệu đã đánh chỉ mục mới có thể đánh dấu bị thay thế")
            old.status, old.legal_status, old.superseded_by = "superseded", "superseded", new_id
            old.effective_to = old.effective_to or new.effective_from
            old.updated_at = new.updated_at = now_iso()
            if old_id not in new.supersedes:
                new.supersedes.append(old_id)
            self._save(records)
            self._audit("supersede", actor, old_id, superseded_by=new_id)

    def mark_indexed(self, document_ids: dict[str, int], embedding_model: str) -> None:
        """Gọi sau khi ghi chỉ mục thành công: approved → indexed, lưu số chunk + phiên bản embedding."""
        with self._lock:
            records = self._load()
            for record in records:
                if record.document_id in document_ids and record.servable:
                    record.status = "indexed"
                    record.chunk_count = document_ids[record.document_id]
                    record.embedding_model_version = embedding_model
                    record.indexed_at = now_iso()
            self._save(records)
        self._audit("index", "system", "*", documents=sorted(document_ids), embedding_model=embedding_model)
