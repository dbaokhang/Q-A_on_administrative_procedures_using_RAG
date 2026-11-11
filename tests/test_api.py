"""Hợp đồng API (mục 2.2.4) và phân quyền endpoint quản trị — với trợ lý giả, không tải mô hình."""

import pytest
from fastapi.testclient import TestClient

from src import api, auth, telemetry
from src.assistant import AssistantResponse


class FakeAssistant:
    pipeline = {"pipeline_version": "test"}

    def __init__(self):
        self.calls = []

    def answer(self, question, user, chat_history="", as_of=None, channel="web"):
        self.calls.append((question, user, chat_history, channel))
        source = {"label": "S1", "chunk_id": "c1", "document_id": "d", "title": "Tên văn bản", "document_number": "1/2020/QH14",
                  "document_type": "law", "location": "Điều 1", "excerpt": "...", "effective_status": "unverified",
                  "effective_status_label": "Chưa xác minh hiệu lực", "effective_from": None, "superseded": False,
                  "source_url": None, "access_scope": "public", "score": 1.0}
        status = "service_unavailable" if "lỗi" in question else "answered_with_sources"
        return AssistantResponse(status, "Trả lời [S1].", "req-0123456789abcdef", intent="khac", sources=[source],
                                 timings={"total_ms": 12.34})


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    monkeypatch.setattr(telemetry, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 0)
    fake = FakeAssistant()
    monkeypatch.setitem(api._state, "assistant", fake)
    test_client = TestClient(api.app)
    test_client.fake = fake
    return test_client


def test_chat_contract(client):
    response = client.post("/chat", json={"question": "Hỏi gì đó", "channel": "web",
                                          "history": [{"role": "user", "content": "trước"}]})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered_with_sources" and body["status_label"] == "Có nguồn phù hợp"
    assert body["correlation_id"] == "req-0123456789abcdef" and body["sources"][0]["document_number"] == "1/2020/QH14"
    assert body["timings_ms"] == {"total_ms": 12.3}
    question, user, history, channel = client.fake.calls[0]
    assert user.role == "citizen" and history == "Người dùng: trước" and channel == "web"


def test_service_error_maps_to_503(client):
    assert client.post("/chat", json={"question": "gây lỗi"}).status_code == 503


def test_client_cannot_choose_its_own_scope(client):
    client.post("/chat", json={"question": "q", "scopes": ["internal"], "role": "admin"})
    assert client.fake.calls[-1][1].scopes == ("public",)


def test_token_sets_role_and_admin_endpoints_are_protected(client, monkeypatch, tmp_path):
    users = tmp_path / "users.json"
    auth.save_user("canbo01", "officer", "matkhau-123", path=users)
    auth.save_user("admin01", "admin", "matkhau-456", path=users)
    monkeypatch.setattr(auth, "USERS_PATH", users)
    monkeypatch.setattr(api, "authenticate", lambda u, p: auth.authenticate(u, p, path=users))

    assert client.post("/auth/login", json={"username": "canbo01", "password": "sai"}).status_code == 401
    officer = client.post("/auth/login", json={"username": "canbo01", "password": "matkhau-123"}).json()["token"]
    client.post("/chat", json={"question": "q"}, headers={"Authorization": f"Bearer {officer}"})
    assert client.fake.calls[-1][1].scopes == ("public", "internal")

    assert client.get("/admin/metrics").status_code == 403                                   # không đăng nhập
    assert client.get("/admin/metrics", headers={"Authorization": f"Bearer {officer}"}).status_code == 403
    assert client.get("/admin/metrics", headers={"Authorization": "Bearer hong"}).status_code == 401
    admin = client.post("/auth/login", json={"username": "admin01", "password": "matkhau-456"}).json()["token"]
    assert client.get("/admin/metrics", headers={"Authorization": f"Bearer {admin}"}).status_code == 200


def test_feedback_validation(client):
    ok = client.post("/feedback", json={"correlation_id": "req-0123456789abcdef", "rating": "not_helpful",
                                        "reason": "outdated", "comment": "gọi 0912345678"})
    assert ok.status_code == 200
    assert client.post("/feedback", json={"correlation_id": "x", "rating": "helpful"}).status_code == 422
    stored = telemetry.read_log("feedback.jsonl")
    assert stored[-1]["reason"] == "outdated" and "0912345678" not in stored[-1]["comment"]
