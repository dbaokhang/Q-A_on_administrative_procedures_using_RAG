"""
Lớp kiểm soát an toàn (mục 2.2.3): đầu vào → nguồn → đầu ra.

Toàn bộ là quy tắc xác định (không gọi mô hình) nên kiểm thử được trong CI. Kiểm tra theo quy tắc không xác
minh được mọi khẳng định ngữ nghĩa; các trường hợp rủi ro cao vẫn cần đánh giá thủ công (RAGAS Faithfulness
và rà soát của cán bộ).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 1000

# ---------------------------------------------------------------- đầu vào

_PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("số điện thoại", re.compile(r"(?<!\d)(?:\+84|0084|0)(?:[\s.-]?\d){9}(?!\d)")),
    # Số định danh cá nhân / CCCD (12 số), CMND (9 số). Số tiền viết có dấu chấm ("1.000.000") không khớp;
    # dấu câu đứng sau số ("CCCD 012345678901,") vẫn khớp.
    ("số định danh", re.compile(r"(?<!\d)(?<!\d[.,])(?:\d{12}|\d{9})(?!\d|[.,]\d)")),
]


@dataclass
class InputCheck:
    ok: bool
    text: str = ""                        # câu hỏi đã chuẩn hóa và che dữ liệu cá nhân
    error: str | None = None              # mã lỗi khi ok = False
    message: str | None = None            # thông điệp cho người dùng
    masked: list[str] = field(default_factory=list)   # loại dữ liệu cá nhân đã che


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Che dữ liệu cá nhân trước khi gửi tới mô hình ngoài hoặc ghi log (BR-07)."""
    found = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(f"[đã che {label}]", text)
    return text, found


def check_input(question: str) -> InputCheck:
    text = unicodedata.normalize("NFC", question or "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return InputCheck(False, error="empty", message="Vui lòng nhập câu hỏi.")
    if len(text) > MAX_QUESTION_CHARS:
        return InputCheck(False, error="too_long",
                          message=f"Câu hỏi dài quá {MAX_QUESTION_CHARS} ký tự, vui lòng rút gọn.")
    masked, found = mask_pii(text)
    return InputCheck(True, text=masked, masked=found)


# ---------------------------------------------------------------- hỏi lại khi thiếu dữ kiện (FR-11, UC-05)

# Chủ đề cụ thể: có một trong các cụm này thì câu hỏi đủ để tra cứu
_TOPIC_TERMS = [
    "khai sinh", "kết hôn", "khai tử", "giám hộ", "nhận cha", "nhận con", "nhận mẹ", "cải chính", "hộ tịch",
    "tình trạng hôn nhân", "độc thân", "chứng thực", "bản sao", "chữ ký", "di chúc", "di sản", "hợp đồng",
    "nuôi con nuôi", "con nuôi", "trích lục", "thường trú", "bảo hiểm", "doanh nghiệp", "công ty", "đất",
    "lao động", "ly hôn", "giao thông", "lái xe", "phạt", "thuế", "xây dựng", "kinh doanh", "hình sự", "thừa kế",
    "thai sản", "nghỉ", "lương", "tạm trú", "điều ", "nghị định", "luật",
]
# Câu hỏi thủ tục chung chung
_GENERIC = re.compile(
    r"\b(hồ sơ|giấy tờ|thủ tục|giấy xác nhận|xác nhận|đăng ký|lệ phí|thời hạn|biểu mẫu|mẫu đơn|tờ khai)\b",
    re.IGNORECASE)

# Gợi ý khi hỏi lại: nhóm thủ tục thường gặp tại Bộ phận Một cửa cấp xã (có trong kho TTHC mẫu)
COMMON_PROCEDURES = [
    "Đăng ký khai sinh",
    "Đăng ký kết hôn",
    "Đăng ký khai tử",
    "Cấp Giấy xác nhận tình trạng hôn nhân",
    "Chứng thực bản sao từ bản chính",
    "Thay đổi, cải chính, bổ sung thông tin hộ tịch",
]


@dataclass
class Clarification:
    question: str
    options: list[str]


def needs_clarification(question: str, chat_history: str = "") -> Clarification | None:
    """Câu hỏi ngắn, chỉ nói "hồ sơ / thủ tục / giấy xác nhận" mà không nêu thủ tục nào → hỏi lại thay vì đoán.

    Có lịch sử hội thoại thì không hỏi lại: câu nối tiếp ("còn lệ phí thì sao?") dựa vào ngữ cảnh trước đó.
    """
    if chat_history.strip():
        return None
    text = question.lower()
    words = re.findall(r"\w+", text)
    if len(words) > 12 or not _GENERIC.search(text) or any(term in text for term in _TOPIC_TERMS):
        return None
    options = COMMON_PROCEDURES
    if "xác nhận" in text:
        options = ["Cấp Giấy xác nhận tình trạng hôn nhân", "Chứng thực chữ ký trong giấy tờ, văn bản",
                   "Chứng thực bản sao từ bản chính"]
    return Clarification(
        question="Bạn muốn hỏi về thủ tục nào? Vui lòng cho biết tên thủ tục (hoặc chọn một gợi ý bên dưới) "
                 "để mình tra cứu đúng thành phần hồ sơ, trình tự và lệ phí.",
        options=options,
    )


_INTERNAL_REQUEST = re.compile(r"(tài liệu|văn bản|hướng dẫn|quy trình|thông tin)\s+(nghiệp vụ\s+)?nội bộ",
                               re.IGNORECASE)


def requests_internal(question: str) -> bool:
    """Người dùng công khai hỏi thẳng tài liệu nội bộ → trả trạng thái "không đủ quyền" (Bảng 2.14)."""
    return bool(_INTERNAL_REQUEST.search(question))


# ---------------------------------------------------------------- nội dung truy xuất (prompt injection)

_INJECTION = re.compile(
    r"(bỏ qua|phớt lờ|quên)\s+(mọi|tất cả|các)?\s*(quy tắc|chỉ dẫn|hướng dẫn|yêu cầu)\s*(trước|ở trên|hệ thống)?"
    r"|ignore (all |any )?(previous|prior|above) (instructions|rules)"
    r"|system prompt|lời nhắc hệ thống|tiết lộ (khóa|mật khẩu|api)",
    re.IGNORECASE)


def injection_suspected(text: str) -> bool:
    """Đoạn tài liệu chứa câu giống chỉ dẫn điều khiển mô hình: vẫn coi là DỮ LIỆU, chỉ gắn cờ để ghi log."""
    return bool(_INJECTION.search(text))


# ---------------------------------------------------------------- kiểm tra trích dẫn sau khi sinh

_MARKER = re.compile(r"\[S(\d+)\]")
_ARTICLE = re.compile(r"\b(?:Điều|Article)\s+(\d+[a-zđ]?)\b", re.IGNORECASE)
_DOC_NUMBER = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9-]*\b")
INSUFFICIENT_TOKEN = "KHÔNG ĐỦ CĂN CỨ"


@dataclass
class CitationCheck:
    passed: bool
    cited: list[int] = field(default_factory=list)       # chỉ số nguồn (1-based) được trích
    issues: list[str] = field(default_factory=list)


def check_citations(answer: str, evidence: list) -> CitationCheck:
    """Mỗi câu trả lời phải trích ít nhất một nguồn [S#] có thật; mọi "Điều N" và số hiệu văn bản nêu trong câu
    trả lời phải xuất hiện trong ngữ cảnh đã truy xuất (mục 2.2.3 d) — tránh tên văn bản / số điều tự sáng tác.

    evidence: danh sách Document (page_content + metadata) theo đúng thứ tự [S1], [S2]...
    """
    issues = []
    cited = sorted({int(n) for n in _MARKER.findall(answer)})
    invalid = [n for n in cited if not 1 <= n <= len(evidence)]
    if invalid:
        issues.append(f"Trích dẫn nguồn không tồn tại: {', '.join(f'[S{n}]' for n in invalid)}")
    valid = [n for n in cited if 1 <= n <= len(evidence)]
    if not valid:
        issues.append("Câu trả lời không gắn mã nguồn [S#] nào")

    context = "\n".join(doc.page_content for doc in evidence)
    known_articles = {m.lower() for m in _ARTICLE.findall(context)}
    for doc in evidence:
        known_articles.update(m.lower() for m in _ARTICLE.findall(doc.metadata.get("article") or ""))
    unknown_articles = sorted({a for a in _ARTICLE.findall(answer) if a.lower() not in known_articles})
    if unknown_articles:
        issues.append(f"Số điều không có trong ngữ cảnh: {', '.join('Điều ' + a for a in unknown_articles)}")

    known_numbers = set(_DOC_NUMBER.findall(context))
    known_numbers.update(doc.metadata.get("document_number") or "" for doc in evidence)
    known_numbers.update(_DOC_NUMBER.findall(" ".join(doc.metadata.get("title") or "" for doc in evidence)))
    unknown_numbers = sorted({n for n in _DOC_NUMBER.findall(answer) if n not in known_numbers})
    if unknown_numbers:
        issues.append(f"Số hiệu văn bản không có trong ngữ cảnh: {', '.join(unknown_numbers)}")
    return CitationCheck(passed=not issues, cited=valid, issues=issues)


def is_insufficient(answer: str) -> bool:
    return INSUFFICIENT_TOKEN.lower() in answer.lower()
