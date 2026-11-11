"""
Giao diện Web (mục 2.4.2): hỏi đáp có nguồn tham khảo + trang quản trị tài liệu cho quản trị viên.

    streamlit run src/app.py
"""

import sys
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth import ANONYMOUS, AuthError, authenticate  # noqa: E402
from src.telemetry import FEEDBACK_REASONS, log_feedback  # noqa: E402

st.set_page_config(page_title="Trợ lý ảo chính quyền cấp xã", page_icon="🏛️", layout="centered")

EXAMPLE_QUESTIONS = [
    "Đăng ký khai sinh cần những giấy tờ gì?",
    "Đăng ký kết hôn mất bao lâu?",
    "Lệ phí chứng thực bản sao từ bản chính là bao nhiêu?",
    "Xin giấy xác nhận tình trạng hôn nhân nộp hồ sơ ở đâu?",
    "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
    "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
]
STATUS_ICONS = {
    "answered_with_sources": "✅", "needs_clarification": "❓", "insufficient_evidence": "📭",
    "access_denied": "🔒", "service_unavailable": "⚠️", "invalid_request": "✏️", "small_talk": "💬",
}


@st.cache_resource(show_spinner="Đang tải mô hình và đồng bộ kho tri thức…")
def assistant_holder() -> dict:
    """Dùng chung cho mọi phiên; quản trị viên đánh chỉ mục lại thì thay phần tử 'assistant'."""
    from src.assistant import build_assistant

    return {"assistant": build_assistant()}


def assistant():
    return assistant_holder()["assistant"]


def _init_state() -> None:
    for key, default in [("messages", []), ("user", ANONYMOUS), ("feedback_sent", set())]:
        st.session_state.setdefault(key, default)


def format_chat_history(messages: list, max_turns: int = 5) -> str:
    lines = []
    for msg in messages[-(max_turns * 2):]:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {msg['content'][:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- thanh bên: vai trò, phiên, kho tri thức

def sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**Vai trò:** {user.role_label}" + (f" · `{user.user_id}`" if user.user_id != "anonymous" else ""))
        if user.user_id == "anonymous":
            with st.expander("Đăng nhập cán bộ / quản trị"):
                with st.form("login", clear_on_submit=True):
                    username = st.text_input("Tên đăng nhập")
                    password = st.text_input("Mật khẩu", type="password")
                    if st.form_submit_button("Đăng nhập"):
                        try:
                            st.session_state.user = authenticate(username, password)
                            st.session_state.messages = []  # phiên mới: không mang hội thoại công khai sang
                            st.rerun()
                        except AuthError as e:
                            st.error(str(e))
        elif st.button("Đăng xuất", width="stretch"):
            st.session_state.user = ANONYMOUS
            st.session_state.messages = []
            st.rerun()

        if st.button("🧹 Cuộc trò chuyện mới", width="stretch",
                     help="Xóa nội dung hội thoại trên màn hình (khuyến nghị khi dùng máy công cộng)"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("**Kho tri thức bạn được tra cứu**")
        titles: dict[str, str] = {}
        for doc in assistant().documents:
            if doc.metadata.get("access_scope", "public") in user.scopes:
                titles.setdefault(doc.metadata.get("document_id"), doc.metadata.get("title"))
        st.markdown("\n".join(f"- {title}" for title in sorted(titles.values())))
        pipeline = assistant().pipeline
        st.caption(f"Phiên bản kho `{pipeline['corpus_version']}` · {pipeline['chunks']} đoạn · "
                   f"{pipeline['embedding_model']} + BM25 ({pipeline['fusion'].upper()}) · {pipeline['llm']}")


# ---------------------------------------------------------------- hiển thị câu trả lời

def _source_card(source: dict) -> None:
    title = source["title"] + (f" — {source['location']}" if source["location"] else "")
    st.markdown(f"**[{source['label']}] {title}**")
    badges = [source["effective_status_label"]]
    if source.get("document_number"):
        badges.insert(0, f"Số hiệu {source['document_number']}")
    if source.get("superseded"):
        badges.append("Phần đã bị thay thế")
    if source.get("access_scope") == "internal":
        badges.append("🔒 Nội bộ")
    st.caption(" · ".join(badges))
    excerpt = source["excerpt"]
    st.text(excerpt[:500] + ("…" if len(excerpt) > 500 else ""))
    if source.get("source_url"):
        st.markdown(f"[Mở nguồn gốc ↗]({source['source_url']}) — *liên kết mở trang bên ngoài trợ lý*")


def render_reply(reply: dict, message_index: int) -> None:
    status = reply["status"]
    icon = STATUS_ICONS.get(status, "")
    if status == "answered_with_sources" or status == "small_talk":
        st.markdown(reply["answer"])
    elif status == "needs_clarification":
        st.info(f"{icon} {reply['answer']}")
        cols = st.columns(2)
        for i, option in enumerate(reply.get("clarification_options", [])):
            if cols[i % 2].button(option, key=f"clarify_{message_index}_{i}", width="stretch"):
                st.session_state.pending_question = f"{reply.get('question', '')} (thủ tục: {option})".strip()
                st.rerun()
    elif status == "insufficient_evidence":
        st.warning(f"{icon} {reply['answer']}")
    else:
        st.error(f"{icon} {reply['answer']}")

    for warning in reply.get("warnings", []):
        st.caption(f"⚠️ {warning}")
    for note in reply.get("notes", []):
        st.info(note, icon="ℹ️")
    if reply.get("sources"):
        with st.expander(f"Nguồn được dùng để trả lời ({len(reply['sources'])})", expanded=False):
            for source in reply["sources"]:
                _source_card(source)
    if reply.get("related_sources"):
        label = "Tài liệu có liên quan" if status != "answered_with_sources" else "Tài liệu liên quan khác (không trích)"
        with st.expander(f"{label} ({len(reply['related_sources'])})", expanded=False):
            for source in reply["related_sources"]:
                _source_card(source)
    if status in ("answered_with_sources", "insufficient_evidence"):
        _feedback_widget(reply)
    st.caption(f"Mã tra cứu: `{reply['correlation_id']}` · {reply.get('total_s', 0):.1f} s")


def _feedback_widget(reply: dict) -> None:
    cid = reply["correlation_id"]
    if cid in st.session_state.feedback_sent:
        st.caption("Cảm ơn bạn đã đánh giá.")
        return
    rating = st.feedback("thumbs", key=f"fb_{cid}")
    if rating == 1:
        log_feedback(cid, "helpful")
        st.session_state.feedback_sent.add(cid)
        st.rerun()
    elif rating == 0:
        with st.form(f"fb_form_{cid}"):
            reason = st.radio("Câu trả lời chưa hữu ích vì:", list(FEEDBACK_REASONS),
                              format_func=FEEDBACK_REASONS.get, horizontal=True)
            comment = st.text_input("Góp ý thêm (không bắt buộc, đừng nhập thông tin cá nhân)")
            if st.form_submit_button("Gửi phản hồi"):
                log_feedback(cid, "not_helpful", reason, comment or None)
                st.session_state.feedback_sent.add(cid)
                st.rerun()


# ---------------------------------------------------------------- trang hỏi đáp

def chat_page() -> None:
    st.title("🏛️ Trợ lý ảo hỗ trợ chính quyền địa phương cấp cơ sở")
    st.caption(
        "Tra cứu thủ tục hành chính cấp xã và văn bản pháp luật, câu trả lời kèm nguồn trích dẫn. "
        "Thông tin chỉ hỗ trợ tra cứu, không thay thế hướng dẫn hay quyết định của cán bộ có thẩm quyền. "
        "🔒 Vui lòng không nhập số định danh cá nhân, số điện thoại, địa chỉ hay thông tin hồ sơ cụ thể."
    )
    messages = st.session_state.messages
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_reply(message["reply"], index)

    examples = st.empty()
    if not messages:
        with examples.container():
            st.markdown("**Câu hỏi thường gặp**")
            cols = st.columns(2)
            for i, question in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 2].button(question, key=f"example_{i}", width="stretch"):
                    st.session_state.pending_question = question

    prompt = st.chat_input("Nhập câu hỏi về thủ tục hành chính hoặc quy định pháp luật…",
                           max_chars=1000) or st.session_state.pop("pending_question", None)
    if not prompt:
        return
    examples.empty()
    with st.chat_message("user"):
        st.write(prompt)
    history = format_chat_history(messages)
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tra cứu tài liệu liên quan…"):
                response = assistant().answer(prompt, st.session_state.user, history, channel="web")
            reply = response.to_dict()
            reply["question"] = prompt
            reply["total_s"] = response.timings.get("total_ms", 0) / 1000
        except Exception:
            traceback.print_exc()
            reply = {"status": "service_unavailable", "correlation_id": "-", "question": prompt,
                     "answer": "Hệ thống đang gặp lỗi tạm thời, vui lòng thử lại sau ít phút."}
        render_reply(reply, len(messages))
    content = reply["answer"] if reply["status"] in ("answered_with_sources", "small_talk") else ""
    messages.append({"role": "assistant", "content": content, "reply": reply})


# ---------------------------------------------------------------- điều hướng

_init_state()
sidebar()
pages = [st.Page(chat_page, title="Hỏi đáp", icon="💬", url_path="hoi-dap", default=True)]
if st.session_state.user.is_admin:
    from src.admin_page import admin_page

    pages.append(st.Page(lambda: admin_page(assistant_holder()), title="Quản trị tài liệu", icon="🗂️",
                         url_path="quan-tri"))
st.navigation(pages).run()
