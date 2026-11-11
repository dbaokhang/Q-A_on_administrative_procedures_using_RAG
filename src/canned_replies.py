"""
Câu trả lời cố định, không cần gọi LLM: lời chào / cảm ơn, và câu hỏi ngoài phạm vi văn bản của hệ thống.
Ngôn ngữ trả lời theo câu hỏi (có dấu tiếng Việt → tiếng Việt; còn lại → song ngữ hoặc tiếng Anh).
"""

import re

try:
    from .legal_glossary import has_vietnamese_diacritics
except ImportError:
    from legal_glossary import has_vietnamese_diacritics

# Chỉ khớp khi CẢ câu là lời chào/cảm ơn — "chào bạn, vượt đèn đỏ phạt bao nhiêu?" vẫn đi qua pipeline
_SUFFIX = r"(?:\s+(?:bạn|nhé|nha|ạ|ad|admin|there|so much))*[\s!.,?~]*"
_GREETING = re.compile(
    r"^\s*(?:xin chào|chào|hello|hi|hey|alo|good (?:morning|afternoon|evening))" + _SUFFIX + r"$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^\s*(?:cảm ơn|cám ơn|thanks|thank you|thank)" + _SUFFIX + r"$", re.IGNORECASE)


def is_small_talk(text: str) -> bool:
    return bool(_GREETING.match(text) or _THANKS.match(text))


def small_talk_reply(text: str) -> str:
    vietnamese = has_vietnamese_diacritics(text)
    if _THANKS.match(text):
        if vietnamese:
            return "Rất vui được hỗ trợ bạn! Bạn còn cần tra cứu thủ tục hay quy định nào khác không?"
        return "You're welcome! · Rất vui được hỗ trợ bạn! Anything else you would like to look up?"
    if vietnamese:
        return "Chào bạn! Bạn cần tra cứu thủ tục hành chính hay quy định pháp luật nào?"
    return ("Xin chào! Bạn cần tra cứu thủ tục hành chính hay quy định pháp luật nào? "
            "· Hello! Which administrative procedure or legal provision can I help you look up?")


# Câu trả lời của LLM kiểu "tài liệu không có thông tin" có thể chỉ là một lần viết lại câu hỏi / sinh câu trả lời
# không may. App không đưa những câu này vào cache dùng chung, để một lần trả lời hỏng không lặp lại cho mọi người.
_NON_ANSWER = re.compile(
    r"không (?:cung cấp|có|đề cập)(?: đến| tới)?(?: bất kỳ| đủ)? thông tin"
    r"|(?:does not|doesn't|do not|don't) (?:provide|contain|mention|include)",
    re.IGNORECASE,
)


def looks_like_non_answer(text: str) -> bool:
    return bool(_NON_ANSWER.search(text))


def out_of_scope_reply(question: str) -> str:
    """Trạng thái "chưa đủ căn cứ" (Bảng 2.14): nêu giới hạn của kho hiện có, không khẳng định "không có quy định"."""
    if has_vietnamese_diacritics(question):
        return (
            "Kho tài liệu hiện có chưa cung cấp căn cứ đủ rõ cho câu hỏi này. Bạn có thể nêu rõ tên thủ tục "
            "hoặc văn bản cần tra cứu, hoặc kiểm tra tại Bộ phận Một cửa của Ủy ban nhân dân cấp xã và "
            "Cổng Dịch vụ công quốc gia (dichvucong.gov.vn)."
        )
    return (
        "The current knowledge base does not contain clear enough grounds to answer this question. Please name the "
        "procedure or document you need, or check with the commune People's Committee one-stop desk or the "
        "National Public Service Portal (dichvucong.gov.vn)."
    )
