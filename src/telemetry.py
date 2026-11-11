"""
Nhật ký truy vấn và phản hồi người dùng (FR-13, FR-16, FR-21, NFR-09).

Mỗi lượt có correlation_id, trạng thái, các đoạn đã dùng, điểm truy xuất, độ trễ từng giai đoạn và phiên bản
pipeline. Mặc định KHÔNG lưu nguyên văn câu hỏi (chỉ lưu băm SHA-256); đặt LOG_QUESTION_TEXT=true để lưu bản đã
che dữ liệu cá nhân, phục vụ phân tích lỗi trong giai đoạn thử nghiệm.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import threading
from pathlib import Path

try:
    from .knowledge.schema import now_iso
except ImportError:
    from knowledge.schema import now_iso

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("LOG_DIR", PROJECT_ROOT / "data" / "logs"))
LOG_QUESTION_TEXT = os.getenv("LOG_QUESTION_TEXT", "false").strip().lower() == "true"
FEEDBACK_REASONS = {
    "wrong_source": "Sai nguồn",
    "missing_condition": "Thiếu điều kiện",
    "outdated": "Thông tin cũ",
    "hard_to_understand": "Khó hiểu",
    "other": "Khác",
}

_lock = threading.Lock()


def _append(name: str, entry: dict, directory: Path | None = None) -> None:
    directory = directory or LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    with _lock, open(directory / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log(name: str, limit: int | None = None, directory: Path | None = None) -> list[dict]:
    path = (directory or LOG_DIR) / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if limit:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines]


def log_request(response, masked_question: str, user_role: str, channel: str, extra: dict | None = None,
                directory: Path | None = None) -> None:
    entry = {
        "at": now_iso(),
        "correlation_id": response.correlation_id,
        "channel": channel,
        "role": user_role,
        "status": response.status,
        "question_sha256": hashlib.sha256(masked_question.encode("utf-8")).hexdigest(),
        "question": masked_question[:300] if LOG_QUESTION_TEXT else None,
        "search_query": response.search_query if LOG_QUESTION_TEXT else None,
        "sources": [s["chunk_id"] for s in response.sources],
        "retrieved": [s["chunk_id"] for s in response.sources + response.related_sources],
        "timings_ms": {k: round(v, 1) for k, v in response.timings.items()},
        "pipeline": response.pipeline,
        **(extra or {}),
    }
    _append("requests.jsonl", entry, directory)


def log_feedback(correlation_id: str, rating: str, reason: str | None = None, comment: str | None = None,
                 directory: Path | None = None) -> dict:
    if rating not in ("helpful", "not_helpful"):
        raise ValueError("rating phải là helpful hoặc not_helpful")
    if reason and reason not in FEEDBACK_REASONS:
        raise ValueError(f"reason không hợp lệ: {reason}")
    try:
        from .guardrails import mask_pii
    except ImportError:
        from guardrails import mask_pii
    entry = {
        "at": now_iso(), "correlation_id": correlation_id, "rating": rating, "reason": reason,
        "comment": mask_pii(comment)[0][:500] if comment else None,
    }
    _append("feedback.jsonl", entry, directory)
    return entry


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summary(directory: Path | None = None) -> dict:
    """Số liệu giám sát cho màn hình quản trị: trạng thái, độ trễ P50/P95, tỷ lệ chưa đủ căn cứ, phản hồi."""
    requests = read_log("requests.jsonl", directory=directory)
    feedback = read_log("feedback.jsonl", directory=directory)
    by_status: dict[str, int] = {}
    for entry in requests:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    totals = [e["timings_ms"].get("total_ms") for e in requests if e.get("timings_ms", {}).get("total_ms")]
    answered = [e for e in requests if e["status"] in ("answered_with_sources", "insufficient_evidence")]
    insufficient = sum(1 for e in answered if e["status"] == "insufficient_evidence")
    reasons: dict[str, int] = {}
    for entry in feedback:
        if entry.get("reason"):
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    return {
        "requests": len(requests),
        "by_status": by_status,
        "latency_ms": {
            "mean": round(statistics.fmean(totals), 1) if totals else None,
            "p50": _percentile(totals, 0.5),
            "p95": _percentile(totals, 0.95),
        },
        "insufficient_rate": round(insufficient / len(answered), 3) if answered else None,
        "feedback": {
            "helpful": sum(1 for e in feedback if e["rating"] == "helpful"),
            "not_helpful": sum(1 for e in feedback if e["rating"] == "not_helpful"),
            "reasons": reasons,
        },
    }
