from src import observability


def test_trace_is_noop_without_keys(monkeypatch):
    monkeypatch.setattr(observability, "ENABLED", False)
    with observability.trace("chat", "question", session_id="abc", tags=["law"]) as trace:
        assert trace.config == {}
        trace.update(output="answer", metadata={"x": 1})  # không lỗi khi không có Langfuse


def test_trace_disabled_for_uploads(monkeypatch):
    monkeypatch.setattr(observability, "ENABLED", True)
    with observability.trace("chat", "question", enabled=False) as trace:
        assert trace.config == {}
