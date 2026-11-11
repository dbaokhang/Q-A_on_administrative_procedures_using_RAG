"""
Phân loại ý định câu hỏi (mục 2.2.2 a) — nhóm theo nhãn của benchmark Vi-CommuneQA (mục 1.3.3):
tra cứu hồ sơ, tra cứu lệ phí, hỏi thời gian, hỏi điều kiện, tra cứu căn cứ pháp lý (+ hỏi trình tự).

Ý định chỉ giúp chọn mục phù hợp của TTHC khi truy xuất (vd. "cần giấy tờ gì" → mục Thành phần hồ sơ), không phải
bằng chứng cho câu trả lời. Quy tắc từ khóa, có thể thay bằng bộ phân loại học máy khi có dữ liệu gán nhãn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (ý định, mục TTHC tương ứng — khóa trong tthc.CHUNK_GROUPS, mẫu nhận diện) — xét theo thứ tự
_RULES = [
    ("tra_cuu_can_cu", "can_cu",
     r"căn cứ (pháp lý|pháp luật)|văn bản nào quy định|quy định ở (văn bản|điều) nào|theo (luật|nghị định) nào"),
    ("tra_cuu_le_phi", "thoi_han_le_phi",
     r"lệ phí|\bphí\b|bao nhiêu tiền|mất bao nhiêu|tốn bao nhiêu|có mất tiền|miễn phí|mức thu"),
    ("hoi_thoi_gian", "thoi_han_le_phi",
     r"bao lâu|mấy ngày|bao nhiêu ngày|thời hạn|khi nào (có|nhận|lấy) kết quả|trong ngày|mất mấy"),
    ("tra_cuu_ho_so", "ho_so",
     r"giấy tờ|thành phần hồ sơ|hồ sơ\b.{0,40}\b(gồm|cần|phải|gì|nào)|(cần|phải) (chuẩn bị|mang|nộp)"
     r"|chuẩn bị (những )?gì|cần những gì|gồm những gì|xuất trình"),
    ("hoi_dieu_kien", "dieu_kien",
     r"điều kiện|yêu cầu gì|có được (làm|đăng ký)|ai được|đủ tuổi|khi nào (thì )?được|trường hợp nào (thì )?được"),
    # Hỏi nơi nộp → mục Thông tin chung (cơ quan thực hiện); hỏi cách làm → mục Trình tự, cách thức
    ("hoi_trinh_tu", "tong_quan", r"nộp ở đâu|làm ở đâu|ở đâu|nơi nộp|cơ quan nào|ai giải quyết"),
    ("hoi_trinh_tu", "trinh_tu",
     r"trình tự|các bước|quy trình|thế nào|ra sao|cách (thức )?(làm|thực hiện|nộp)"
     r"|trực tuyến|online|qua mạng|bưu chính|ủy quyền"),
]
_COMPILED = [(name, group, re.compile(pattern, re.IGNORECASE)) for name, group, pattern in _RULES]

INTENT_LABELS = {
    "tra_cuu_ho_so": "Tra cứu hồ sơ",
    "tra_cuu_le_phi": "Tra cứu lệ phí",
    "hoi_thoi_gian": "Hỏi thời gian",
    "hoi_dieu_kien": "Hỏi điều kiện",
    "tra_cuu_can_cu": "Tra cứu căn cứ pháp lý",
    "hoi_trinh_tu": "Hỏi trình tự, cách thức",
    "khac": "Khác",
}


@dataclass(frozen=True)
class Intent:
    name: str
    procedure_field: str | None   # mục TTHC nên ưu tiên khi truy xuất

    @property
    def label(self) -> str:
        return INTENT_LABELS[self.name]


def classify_intent(question: str) -> Intent:
    for name, group, pattern in _COMPILED:
        if pattern.search(question):
            return Intent(name, group)
    return Intent("khac", None)
