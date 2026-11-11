"""
Bộ lọc truy xuất theo phạm vi quyền và hiệu lực (BR-03, BR-04, BR-10; mục 2.3.2 f).

Cùng một SearchFilter dùng cho cả nhánh sparse (BM25, lọc bằng allows()) và nhánh dense (Qdrant: to_qdrant(),
FAISS: allows()) để lọc quyền chạy TRƯỚC khi dựng ngữ cảnh, ở cả hai nhánh. Phạm vi quyền do backend xác định
từ phiên đăng nhập (auth.py), không nhận từ client.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchFilter:
    scopes: tuple[str, ...] = ("public",)
    as_of: dt.date = field(default_factory=dt.date.today)
    include_superseded: bool = False   # câu hỏi nhắc thẳng tới văn bản cũ (vd. "NĐ 100/2019")

    @property
    def as_of_ord(self) -> int:
        return self.as_of.toordinal()

    def allows(self, meta: dict) -> bool:
        if meta.get("access_scope", "public") not in self.scopes:
            return False
        start, end = meta.get("effective_from_ord"), meta.get("effective_to_ord")
        if start is not None and start > self.as_of_ord:
            return False          # chưa có hiệu lực tại mốc tra cứu
        if end is not None and end <= self.as_of_ord:
            return False          # đã hết hiệu lực
        if meta.get("superseded") and not self.include_superseded:
            since = meta.get("superseded_from_ord")
            if since is None or since <= self.as_of_ord:
                return False      # phần đã bị thay thế tại mốc tra cứu
        return True

    def to_qdrant(self):
        from qdrant_client import models

        as_of = self.as_of_ord
        must_not = [
            models.FieldCondition(key="effective_from_ord", range=models.Range(gt=as_of)),
            models.FieldCondition(key="effective_to_ord", range=models.Range(lte=as_of)),
        ]
        if not self.include_superseded:
            superseded = models.FieldCondition(key="superseded", match=models.MatchValue(value=True))
            must_not += [
                models.Filter(must=[superseded, models.FieldCondition(
                    key="superseded_from_ord", range=models.Range(lte=as_of))]),
                # không rõ ngày bị thay thế → coi như đã bị thay thế (giống allows())
                models.Filter(must=[superseded, models.IsNullCondition(
                    is_null=models.PayloadField(key="superseded_from_ord"))]),
            ]
        return models.Filter(
            must=[models.FieldCondition(key="access_scope", match=models.MatchAny(any=list(self.scopes)))],
            must_not=must_not,
        )
