"""
Màn hình quản trị tri thức (mục 2.4.3): tài liệu, trạng thái, metadata hiệu lực, đánh chỉ mục, giám sát, nhật ký.

Mọi thao tác thay đổi đều ghi audit log (BR-08); tài liệu mới nạp ở trạng thái "chờ kiểm tra" và chỉ được dùng
để trả lời sau khi được duyệt và đánh chỉ mục (BR-01). Trang chỉ hiển thị khi phiên có vai trò admin; API quản trị
cũng tự kiểm tra quyền (không dựa vào việc ẩn trang).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from src.document_upload import MAX_FILE_MB, UploadError, extract_text, validate_uploads
from src.knowledge.registry import PROJECT_ROOT, DocumentRegistry, RegistryError
from src.knowledge.schema import (ACCESS_SCOPES, ALLOWED_TRANSITIONS, DOCUMENT_TYPES, LEGAL_STATUSES,
                                  WORKFLOW_STATUSES, DocumentRecord)
from src.knowledge.tthc import load_procedures
from src.telemetry import FEEDBACK_REASONS, read_log, summary

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ACTION_LABELS = {
    "approved": "✅ Duyệt", "needs_review": "↩️ Đưa về chờ kiểm tra", "withdrawn": "⛔ Rút khỏi kho",
    "received": "🔁 Tiếp nhận lại", "failed": "❌ Đánh dấu lỗi",
}


def _slug(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:60] or "tai-lieu"


def _documents_tab(registry: DocumentRegistry, user_id: str) -> None:
    records = registry.all()
    table = pd.DataFrame([{
        "Mã": r.document_id, "Tên": r.title, "Loại": DOCUMENT_TYPES[r.document_type],
        "Phạm vi": ACCESS_SCOPES[r.access_scope], "Quy trình": WORKFLOW_STATUSES[r.status],
        "Hiệu lực": LEGAL_STATUSES[r.legal_status], "Hiệu lực từ": r.effective_from, "Số đoạn": r.chunk_count,
    } for r in records])
    st.dataframe(table, hide_index=True, width="stretch")

    selected_id = st.selectbox("Chọn tài liệu", [r.document_id for r in records],
                               format_func=lambda i: next(f"{r.title} ({i})" for r in records if r.document_id == i))
    record = registry.get(selected_id)
    st.markdown(f"**Trạng thái:** {WORKFLOW_STATUSES[record.status]} · **Phiên bản:** `{record.version_id}` · "
                f"**Duyệt bởi:** {record.reviewed_by or '—'} · **Đánh chỉ mục:** {record.indexed_at or '—'} "
                f"({record.embedding_model_version or '—'})")

    cols = st.columns(4)
    for i, target in enumerate(t for t in ACTION_LABELS if t in ALLOWED_TRANSITIONS[record.status]):
        if cols[i % 4].button(ACTION_LABELS[target], key=f"act_{target}", width="stretch"):
            _apply(lambda: registry.set_status(record.document_id, target, user_id))

    replaceable = [r for r in records if r.document_id != record.document_id and r.status == "indexed"]
    if record.servable and replaceable:
        with st.expander("Đánh dấu tài liệu khác bị tài liệu này thay thế"):
            old_id = st.selectbox("Tài liệu bị thay thế", [r.document_id for r in replaceable])
            if st.button("Xác nhận thay thế", type="primary"):
                _apply(lambda: registry.supersede(old_id, record.document_id, user_id))

    with st.expander("Sửa metadata hiệu lực và trích dẫn"):
        with st.form("metadata"):
            title = st.text_input("Tên tài liệu", record.title)
            number = st.text_input("Số hiệu", record.document_number or "")
            body = st.text_input("Cơ quan ban hành", record.issuing_body or "")
            c1, c2, c3 = st.columns(3)
            issued = c1.text_input("Ngày ban hành (YYYY-MM-DD)", record.issued_date or "")
            start = c2.text_input("Hiệu lực từ (YYYY-MM-DD)", record.effective_from or "")
            end = c3.text_input("Hết hiệu lực (YYYY-MM-DD, để trống nếu chưa xác định)", record.effective_to or "")
            legal = st.selectbox("Trạng thái hiệu lực", list(LEGAL_STATUSES), format_func=LEGAL_STATUSES.get,
                                 index=list(LEGAL_STATUSES).index(record.legal_status))
            scope = st.selectbox("Phạm vi truy cập", list(ACCESS_SCOPES), format_func=ACCESS_SCOPES.get,
                                 index=list(ACCESS_SCOPES).index(record.access_scope))
            url = st.text_input("Liên kết nguồn đã xác minh", record.source_url or "")
            notes = st.text_area("Lưu ý hiển thị kèm nguồn", record.notes or "")
            if st.form_submit_button("Lưu metadata"):
                _apply(lambda: registry.update_metadata(
                    record.document_id, user_id, title=title, document_number=number or None,
                    issuing_body=body or None, issued_date=issued or None, effective_from=start or None,
                    effective_to=end or None, legal_status=legal, access_scope=scope, source_url=url or None,
                    notes=notes or None))


def _apply(action) -> None:
    try:
        action()
        st.success("Đã lưu. Bấm \"Đánh chỉ mục lại\" để áp dụng cho trợ lý.")
        st.rerun()
    except (RegistryError, ValueError) as e:
        st.error(str(e))


def _upload_tab(registry: DocumentRegistry, user_id: str) -> None:
    st.caption(f"PDF, DOCX, TXT, MD hoặc tệp JSON TTHC (≤ {MAX_FILE_MB} MB). Tài liệu mới ở trạng thái "
               "\"Chờ kiểm tra\" — chỉ được dùng để trả lời sau khi duyệt và đánh chỉ mục.")
    with st.form("upload", clear_on_submit=True):
        file = st.file_uploader("Tệp tài liệu", type=["pdf", "docx", "txt", "md", "json"])
        title = st.text_input("Tên tài liệu (bắt buộc)")
        c1, c2 = st.columns(2)
        doc_type = c1.selectbox("Loại", list(DOCUMENT_TYPES), format_func=DOCUMENT_TYPES.get)
        scope = c2.selectbox("Phạm vi truy cập", list(ACCESS_SCOPES), format_func=ACCESS_SCOPES.get)
        number = c1.text_input("Số hiệu")
        body = c2.text_input("Cơ quan ban hành")
        issued = c1.date_input("Ngày ban hành", value=None)
        start = c2.date_input("Hiệu lực từ", value=None)
        url = st.text_input("Liên kết nguồn")
        submitted = st.form_submit_button("Tiếp nhận tài liệu")
    if not submitted:
        return
    if not file or not title.strip():
        st.error("Cần chọn tệp và nhập tên tài liệu.")
        return
    data = file.getvalue()
    suffix = Path(file.name).suffix.lower()
    try:
        if suffix == ".json":
            if doc_type != "administrative_procedure":
                raise UploadError("Tệp JSON chỉ dùng cho loại Thủ tục hành chính.")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            probe = UPLOAD_DIR / f".probe-{file.name}"
            probe.write_bytes(data)
            procedures = load_procedures(probe)
            probe.unlink()
            preview = f"{len(procedures)} thủ tục: " + "; ".join(p["title"] for p in procedures[:5])
        else:
            validate_uploads([(file.name, len(data))])
            text = extract_text(file.name, data)
            preview = f"{len(text):,} ký tự. Đoạn đầu: {text[:300]}"
        doc_id = f"{_slug(title)}-{dt.date.today():%Y%m%d}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / f"{doc_id}{suffix}"
        path.write_bytes(data)
        record = DocumentRecord(
            document_id=doc_id, title=title.strip(), document_type=doc_type,
            file_path=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), document_number=number or None,
            issuing_body=body or None, issued_date=issued.isoformat() if issued else None,
            effective_from=start.isoformat() if start else None, access_scope=scope, source_url=url or None,
            status="needs_review", source_note=f"Tải lên bởi {user_id}: {file.name}",
        )
        registry.register(record, user_id)
        st.success(f"Đã tiếp nhận {doc_id} (chờ kiểm tra). {preview}")
    except (UploadError, RegistryError, ValueError, json.JSONDecodeError, KeyError) as e:
        st.error(f"Không tiếp nhận được tài liệu: {e}")


def _monitor_tab() -> None:
    stats = summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lượt hỏi", stats["requests"])
    c2.metric("Độ trễ P50 (s)", f"{stats['latency_ms']['p50'] / 1000:.1f}" if stats["latency_ms"]["p50"] else "—")
    c3.metric("Độ trễ P95 (s)", f"{stats['latency_ms']['p95'] / 1000:.1f}" if stats["latency_ms"]["p95"] else "—")
    rate = stats["insufficient_rate"]
    c4.metric("Tỷ lệ chưa đủ căn cứ", f"{rate:.0%}" if rate is not None else "—")
    if stats["by_status"]:
        st.bar_chart(pd.Series(stats["by_status"], name="Số lượt"), horizontal=True)
    fb = stats["feedback"]
    st.markdown(f"**Phản hồi:** 👍 {fb['helpful']} · 👎 {fb['not_helpful']} — "
                + ", ".join(f"{FEEDBACK_REASONS[k]}: {v}" for k, v in fb["reasons"].items()))
    recent = read_log("requests.jsonl", limit=50)
    if recent:
        st.markdown("**Lượt hỏi gần đây** (không lưu nguyên văn câu hỏi trừ khi bật LOG_QUESTION_TEXT)")
        st.dataframe(pd.DataFrame([{
            "Thời điểm": e["at"], "Mã": e["correlation_id"], "Kênh": e["channel"], "Vai trò": e["role"],
            "Trạng thái": e["status"], "Ý định": e.get("intent"), "Nguồn": len(e["sources"]),
            "Tổng (ms)": e["timings_ms"].get("total_ms"), "Câu hỏi": e.get("question"),
        } for e in reversed(recent)]), hide_index=True, width="stretch")
    feedback = read_log("feedback.jsonl", limit=50)
    if feedback:
        st.markdown("**Phản hồi gần đây**")
        st.dataframe(pd.DataFrame(list(reversed(feedback))), hide_index=True, width="stretch")


def admin_page(holder: dict) -> None:
    user = st.session_state.user
    if not user.is_admin:
        st.error("Trang này chỉ dành cho quản trị viên.")
        return
    st.title("🗂️ Quản trị kho tri thức")
    registry = DocumentRegistry()
    pipeline = holder["assistant"].pipeline
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.caption(f"Đang phục vụ: phiên bản kho `{pipeline['corpus_version']}`, {pipeline['chunks']} đoạn, "
               f"embedding `{pipeline['embedding_model']}`, vector `{pipeline['vector_backend']}`")
    if c2.button("🔄 Đánh chỉ mục lại", type="primary", width="stretch"):
        from src.assistant import build_assistant

        try:
            with st.spinner("Đang đồng bộ chỉ mục (chỉ embed đoạn mới)…"):
                holder["assistant"] = build_assistant(previous=holder["assistant"])
            st.success(f"Đã cập nhật: {holder['assistant'].pipeline['chunks']} đoạn.")
        except Exception as e:
            traceback.print_exc()
            st.error(f"Đánh chỉ mục thất bại, trợ lý cần khởi động lại: {e}")

    tabs = st.tabs(["Tài liệu", "Nạp tài liệu mới", "Giám sát", "Nhật ký thao tác"])
    with tabs[0]:
        _documents_tab(registry, user.user_id)
    with tabs[1]:
        _upload_tab(registry, user.user_id)
    with tabs[2]:
        _monitor_tab()
    with tabs[3]:
        log = registry.audit_log(200)
        if log:
            st.dataframe(pd.DataFrame(log), hide_index=True, width="stretch")
        else:
            st.caption("Chưa có thao tác nào.")
