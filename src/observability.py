"""
Giám sát bằng Langfuse (tuỳ chọn): mỗi câu hỏi là một trace gồm bước viết lại câu hỏi và sinh câu trả lời
(token, độ trễ), cộng metadata: trạng thái trả lời, correlation_id, ý định, điểm rerank cao nhất.
Nhật ký nội bộ luôn bật nằm ở telemetry.py; Langfuse chỉ là lớp quan sát bổ sung.

Chỉ bật khi có LANGFUSE_PUBLIC_KEY và LANGFUSE_SECRET_KEY (LANGFUSE_BASE_URL mặc định là Langfuse Cloud EU).
Không có key, hoặc chưa cài thư viện langfuse → mọi hàm ở đây là no-op.
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if ENABLED:
    try:
        from langfuse import get_client, propagate_attributes
        from langfuse.langchain import CallbackHandler
    except ImportError:
        print("LANGFUSE_* keys are set but the langfuse package is not installed — tracing disabled.")
        ENABLED = False


class Trace:
    """Kết quả của trace(): config truyền vào LangChain, và update() để ghi metadata / output."""

    def __init__(self, span=None):
        self._span = span
        self.config = {"callbacks": [CallbackHandler()]} if span is not None else {}

    def update(self, **fields: Any) -> None:
        if self._span is not None:
            self._span.update(**fields)


@contextmanager
def trace(name: str, question: str, *, session_id: str | None = None,
          tags: list[str] | None = None, enabled: bool = True) -> Iterator[Trace]:
    if not (ENABLED and enabled):
        yield Trace()
        return
    with ExitStack() as stack:
        try:
            span = stack.enter_context(
                get_client().start_as_current_observation(name=name, as_type="chain", input=question))
            stack.enter_context(propagate_attributes(session_id=session_id, tags=tags or []))
        except Exception as e:  # lỗi của Langfuse không được làm hỏng câu trả lời
            print(f"Langfuse trace failed to start: {e}")
            span = None
        yield Trace(span)
