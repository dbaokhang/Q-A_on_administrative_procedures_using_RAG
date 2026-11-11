"""
Schema metadata của kho tri thức (mục 2.3.1–2.3.2 báo cáo).

Hai loại trạng thái khác nhau, không gộp làm một:
- status: trạng thái quy trình quản trị (received → needs_review → approved → indexed ...). Chỉ tài liệu
  approved/indexed mới được đưa vào chỉ mục trả lời (BR-01).
- legal_status: trạng thái hiệu lực pháp lý đã được xác minh hay chưa. Tài liệu được duyệt đưa vào kho vẫn có
  thể "unverified" — giao diện phải hiển thị điều đó thay vì coi là còn hiệu lực.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field, fields

DOCUMENT_TYPES = {
    "law": "Văn bản quy phạm pháp luật",
    "administrative_procedure": "Thủ tục hành chính",
    "internal_guide": "Tài liệu nghiệp vụ nội bộ",
}

ACCESS_SCOPES = {
    "public": "Công khai",
    "internal": "Nội bộ (cán bộ được phân quyền)",
}

LEGAL_STATUSES = {
    "verified_current": "Đã xác minh còn hiệu lực",
    "unverified": "Chưa xác minh hiệu lực",
    "amended": "Còn hiệu lực, đã được sửa đổi, bổ sung một phần",
    "partially_superseded": "Bị thay thế một phần",
    "superseded": "Đã bị thay thế",
    "expired": "Hết hiệu lực",
}

WORKFLOW_STATUSES = {
    "received": "Đã tiếp nhận",
    "extracting": "Đang trích xuất",
    "needs_review": "Chờ kiểm tra",
    "approved": "Đã duyệt (chờ đánh chỉ mục)",
    "indexed": "Đã đánh chỉ mục",
    "superseded": "Đã bị thay thế",
    "withdrawn": "Đã rút khỏi kho",
    "failed": "Lỗi xử lý",
}

# Chuyển trạng thái hợp lệ (mục 2.3.1 f): chỉ tài liệu approved mới được đánh chỉ mục
ALLOWED_TRANSITIONS = {
    "received": {"extracting", "needs_review", "failed", "withdrawn"},
    "extracting": {"needs_review", "failed"},
    "needs_review": {"approved", "withdrawn", "failed"},
    "approved": {"indexed", "needs_review", "withdrawn"},
    "indexed": {"indexed", "needs_review", "superseded", "withdrawn"},
    "superseded": {"needs_review"},
    "withdrawn": {"needs_review"},
    "failed": {"received"},
}

# Trạng thái được dùng để trả lời
SERVABLE_STATUSES = {"approved", "indexed"}

_ID_NAMESPACE = uuid.UUID("6f1f8a52-4d1e-4c55-9a7e-2f3c1d0b7e11")


def today() -> dt.date:
    return dt.date.today()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_date(value: str | dt.date | None) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def date_ordinal(value: str | dt.date | None) -> int | None:
    """Ngày → số nguyên để lọc theo khoảng trong vector database."""
    parsed = parse_date(value)
    return parsed.toordinal() if parsed else None


def stable_chunk_id(version_id: str, index: int) -> str:
    """Khóa ổn định theo phiên bản + thứ tự đoạn → nạp lại không tạo chunk trùng (idempotent)."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{version_id}#{index}"))


@dataclass
class DocumentRecord:
    """Một phiên bản tài liệu trong kho (gộp Document + DocumentVersion của mô hình logic, mục 2.3.2 a)."""

    document_id: str
    title: str
    document_type: str
    file_path: str
    file_hash: str = ""
    version_id: str = ""
    document_number: str | None = None
    issuing_body: str | None = None
    issued_date: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    legal_status: str = "unverified"
    access_scope: str = "public"
    source_url: str | None = None
    source_note: str | None = None
    field_area: str | None = None          # lĩnh vực (Hộ tịch, Chứng thực, Giao thông...)
    status: str = "received"
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    notes: str | None = None               # lưu ý hiển thị kèm nguồn (vd. đã bị thay thế một phần)
    reviewed_by: str | None = None
    ingested_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    indexed_at: str | None = None
    embedding_model_version: str | None = None
    chunk_count: int | None = None

    def __post_init__(self) -> None:
        if self.document_type not in DOCUMENT_TYPES:
            raise ValueError(f"document_type không hợp lệ: {self.document_type!r}")
        if self.access_scope not in ACCESS_SCOPES:
            raise ValueError(f"access_scope không hợp lệ: {self.access_scope!r}")
        if self.legal_status not in LEGAL_STATUSES:
            raise ValueError(f"legal_status không hợp lệ: {self.legal_status!r}")
        if self.status not in WORKFLOW_STATUSES:
            raise ValueError(f"status không hợp lệ: {self.status!r}")
        for name in ("issued_date", "effective_from", "effective_to"):
            value = getattr(self, name)
            if value:
                parse_date(value)  # định dạng ISO 8601, sai thì báo lỗi ngay khi nhập
        if not self.version_id and self.file_hash:
            self.version_id = f"{self.document_id}@{self.file_hash[:12]}"

    @property
    def servable(self) -> bool:
        return self.status in SERVABLE_STATUSES

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def chunk_metadata(self) -> dict:
        """Metadata chung gắn vào mọi chunk của tài liệu (mục 2.3.2 b)."""
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "title": self.title,
            "law": self.title,  # nhãn hiển thị / trích dẫn
            "document_number": self.document_number,
            "document_type": self.document_type,
            "issuing_body": self.issuing_body,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "effective_from_ord": date_ordinal(self.effective_from),
            "effective_to_ord": date_ordinal(self.effective_to),
            "legal_status": self.legal_status,
            "access_scope": self.access_scope,
            "source_url": self.source_url,
            "field_area": self.field_area,
        }
