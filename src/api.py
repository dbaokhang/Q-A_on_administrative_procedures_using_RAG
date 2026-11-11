"""
API hỏi đáp và quản trị (hợp đồng API mục 2.2.4).

    uvicorn src.api:app            # tài liệu tương tác: http://localhost:8000/docs

Mọi kênh (Web, Zalo OA adapter, kiosk, voicebot) dùng chung POST /chat. Vai trò lấy từ token đã xác thực
(Authorization: Bearer ...); không có token = người dân (chỉ kho công khai).
"""

import datetime as dt
import os
import threading
import time
from collections import defaultdict, deque
from typing import List, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from . import telemetry
    from .auth import ANONYMOUS, AuthError, UserContext, authenticate, issue_token, verify_token
    from .knowledge.registry import DocumentRegistry, RegistryError
except ImportError:
    import telemetry
    from auth import ANONYMOUS, AuthError, UserContext, authenticate, issue_token, verify_token
    from knowledge.registry import DocumentRegistry, RegistryError

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))  # mỗi địa chỉ IP, kênh công khai

app = FastAPI(title="Trợ lý ảo hỗ trợ chính quyền địa phương cấp cơ sở — API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"assistant": None, "lock": threading.Lock()}
_hits: dict[str, deque] = defaultdict(deque)


def get_assistant():
    """Nạp trợ lý ở lần gọi đầu (tải mô hình, đồng bộ chỉ mục) — /health không phải chờ."""
    with _state["lock"]:
        if _state["assistant"] is None:
            try:
                from .assistant import build_assistant
            except ImportError:
                from assistant import build_assistant
            _state["assistant"] = build_assistant()
        return _state["assistant"]


def current_user(authorization: str | None = Header(default=None)) -> UserContext:
    if not authorization:
        return ANONYMOUS
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization phải có dạng 'Bearer <token>'")
    try:
        return verify_token(token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from None


def admin_user(user: UserContext = Depends(current_user)) -> UserContext:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Chức năng chỉ dành cho quản trị viên")
    return user


def rate_limit(request: Request, user: UserContext = Depends(current_user)) -> None:
    if not RATE_LIMIT_PER_MINUTE or user.role != "citizen":
        return
    key = request.client.host if request.client else "unknown"
    now = time.time()
    window = _hits[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu, vui lòng thử lại sau ít phút")
    window.append(now)


# ---------------------------------------------------------------- mô hình dữ liệu

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(default=None, max_length=100)
    question: str = Field(max_length=4000)  # giới hạn nghiệp vụ (1000 ký tự) kiểm tra ở guardrails
    channel: Literal["web", "zalo", "kiosk", "voice", "api"] = "api"
    locale: str = "vi-VN"
    requested_at: dt.datetime | None = None
    as_of: dt.date | None = Field(default=None, description="Mốc thời gian tra cứu hiệu lực (mặc định: hôm nay)")
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)


class SourceOut(BaseModel):
    label: str
    chunk_id: str | None
    document_id: str | None
    title: str
    document_number: str | None
    document_type: str | None
    location: str
    excerpt: str
    effective_status: str
    effective_status_label: str
    effective_from: str | None
    superseded: bool
    source_url: str | None
    access_scope: str
    score: float | None


class ChatResponse(BaseModel):
    answer: str
    status: str
    status_label: str
    intent: str | None
    sources: List[SourceOut]
    related_sources: List[SourceOut]
    clarification_options: List[str]
    notes: List[str]
    warnings: List[str]
    search_query: str
    correlation_id: str
    timings_ms: dict


class LoginRequest(BaseModel):
    username: str
    password: str


class FeedbackRequest(BaseModel):
    correlation_id: str = Field(pattern=r"^req-[0-9a-f]{16}$")
    rating: Literal["helpful", "not_helpful"]
    reason: Literal["wrong_source", "missing_condition", "outdated", "hard_to_understand", "other"] | None = None
    comment: str | None = Field(default=None, max_length=1000)


class StatusChange(BaseModel):
    status: str
    note: str | None = None


def format_history(history: List[ChatTurn], max_turns: int = 5) -> str:
    """Cùng định dạng lịch sử hội thoại với giao diện Web."""
    lines = []
    for turn in history[-(max_turns * 2):]:
        role = "Người dùng" if turn.role == "user" else "Trợ lý"
        lines.append(f"{role}: {turn.content[:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- hỏi đáp

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(request: ChatRequest, user: UserContext = Depends(current_user)):
    # def (không async): xử lý đồng bộ, FastAPI đưa vào threadpool thay vì chặn event loop
    result = get_assistant().answer(request.question, user, format_history(request.history),
                                    as_of=request.as_of, channel=request.channel)
    payload = result.to_dict()
    payload["timings_ms"] = {k: round(v, 1) for k, v in payload.pop("timings").items()}
    payload.pop("pipeline")
    code = {"service_unavailable": 503, "invalid_request": 400}.get(result.status, 200)
    return JSONResponse(ChatResponse(**payload).model_dump(), status_code=code)


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    entry = telemetry.log_feedback(request.correlation_id, request.rating, request.reason, request.comment)
    return {"ok": True, "recorded_at": entry["at"]}


@app.post("/auth/login")
def login(request: LoginRequest):
    try:
        user = authenticate(request.username, request.password)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from None
    token, expires = issue_token(user)
    return {"token": token, "role": user.role, "role_label": user.role_label, "expires_at": expires}


@app.get("/health")
def health():
    assistant = _state["assistant"]
    return {"status": "ok", "ready": assistant is not None, "pipeline": assistant.pipeline if assistant else None}


# ---------------------------------------------------------------- quản trị (FR-14, FR-15, FR-21, FR-22)

@app.get("/admin/documents")
def list_documents(_: UserContext = Depends(admin_user)):
    return [record.to_dict() for record in DocumentRegistry().all()]


@app.post("/admin/documents/{document_id}/status")
def change_status(document_id: str, change: StatusChange, user: UserContext = Depends(admin_user)):
    try:
        record = DocumentRegistry().set_status(document_id, change.status, user.user_id, change.note)
    except RegistryError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return record.to_dict()


@app.post("/admin/reindex")
def reindex(_: UserContext = Depends(admin_user)):
    """Đồng bộ chỉ mục với registry (tài liệu mới duyệt / bị rút) rồi nạp lại trợ lý."""
    try:
        from .assistant import build_assistant
    except ImportError:
        from assistant import build_assistant
    with _state["lock"]:
        _state["assistant"] = build_assistant(previous=_state["assistant"])
        return _state["assistant"].pipeline


@app.get("/admin/metrics")
def metrics(_: UserContext = Depends(admin_user)):
    return telemetry.summary()


@app.get("/admin/audit")
def audit(limit: int = 100, _: UserContext = Depends(admin_user)):
    return DocumentRegistry().audit_log(limit)
