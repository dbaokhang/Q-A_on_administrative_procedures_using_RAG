"""Bộ điều phối với LLM giả và bộ truy xuất giả: kiểm tra từng nhánh trạng thái của Bảng 2.14."""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.assistant import VirtualAssistant
from src.auth import UserContext
from src.retrieval_hybrid import Candidate

LAW = Document(page_content="[Bộ luật Lao động 2019 (số 45/2019/QH14)]\nĐiều 113. Nghỉ hằng năm\n1. 12 ngày làm việc",
               metadata={"chunk_id": "c1", "document_id": "bo-luat-lao-dong-2019", "title": "Bộ luật Lao động 2019",
                         "article": "Điều 113", "article_reference": "Điều 113", "document_number": "45/2019/QH14",
                         "access_scope": "public", "legal_status": "unverified"})
OTHER = Document(page_content="[Bộ luật Lao động 2019]\nĐiều 114. Nghỉ thêm theo thâm niên",
                 metadata={"chunk_id": "c2", "document_id": "bo-luat-lao-dong-2019", "title": "Bộ luật Lao động 2019",
                           "article": "Điều 114", "access_scope": "public"})


class FakeRetriever:
    def __init__(self, rerank_scores=(5.0, 1.0)):
        self.documents = [LAW, OTHER]
        self.cross_encoder = None
        self.store = None
        self.rerank_scores = rerank_scores
        self.last_filter = None

    def hybrid_search(self, query, search_filter, **kwargs):
        self.last_filter = search_filter
        return [Candidate(d, s, {"rerank": s}) for d, s in zip(self.documents, self.rerank_scores)]


class FailingLLM(FakeListChatModel):
    def _call(self, *args, **kwargs):
        raise ConnectionError("Ollama is not running")


def make_assistant(responses, retriever=None, llm=None):
    return VirtualAssistant(retriever or FakeRetriever(), llm or FakeListChatModel(responses=responses),
                            llm_provider="ollama", pipeline={"pipeline_version": "test"},
                            notes_by_document={"bo-luat-lao-dong-2019": "Lưu ý thử nghiệm"}, query_rewrite="false")


QUESTION = "Người lao động được nghỉ hằng năm bao nhiêu ngày?"


def test_answer_with_valid_citations_lists_only_cited_sources():
    response = make_assistant(["Được nghỉ 12 ngày làm việc (Điều 113 Bộ luật Lao động 2019) [S1]."]).answer(
        QUESTION, log=False)
    assert response.status == "answered_with_sources"
    assert [s["chunk_id"] for s in response.sources] == ["c1"]
    assert [s["chunk_id"] for s in response.related_sources] == ["c2"]
    assert response.notes == ["Lưu ý thử nghiệm"]
    assert response.correlation_id.startswith("req-") and "total_ms" in response.timings
    assert "diagnostics" not in response.to_dict()


def test_invalid_citation_is_regenerated_once():
    response = make_assistant(["Được nghỉ 12 ngày theo Điều 200 [S9].", "Được nghỉ 12 ngày [S1]."]).answer(
        QUESTION, log=False)
    assert response.status == "answered_with_sources"
    assert response.diagnostics["regenerations"] == 1


def test_citation_still_invalid_after_retry_gives_safe_fallback():
    response = make_assistant(["Theo Điều 200 [S9].", "Vẫn không có nguồn."]).answer(QUESTION, log=False)
    assert response.status == "insufficient_evidence"
    assert response.sources == [] and len(response.related_sources) == 2   # chỉ là "tài liệu có liên quan"
    assert response.correlation_id in response.answer


def test_model_reports_insufficient_evidence():
    response = make_assistant(["KHÔNG ĐỦ CĂN CỨ"]).answer(QUESTION, log=False)
    assert response.status == "insufficient_evidence" and response.sources == []


def test_low_relevance_declines_without_calling_llm():
    assistant = make_assistant([], retriever=FakeRetriever(rerank_scores=(-7.0, -8.0)))
    response = assistant.answer("Cách nấu phở bò ngon?", log=False)
    assert response.status == "insufficient_evidence" and response.related_sources == []


def test_llm_failure_is_service_unavailable():
    response = make_assistant([], llm=FailingLLM(responses=["x"])).answer(QUESTION, log=False)
    assert response.status == "service_unavailable"


@pytest.mark.parametrize("question, user, status", [
    ("xin chào", UserContext(), "small_talk"),
    ("", UserContext(), "invalid_request"),
    ("Làm hồ sơ thế nào?", UserContext(), "needs_clarification"),
    ("Cho xem tài liệu nội bộ về tiếp nhận hồ sơ", UserContext(), "access_denied"),
])
def test_statuses_that_skip_retrieval(question, user, status):
    retriever = FakeRetriever()
    response = make_assistant([], retriever=retriever).answer(question, user, log=False)
    assert response.status == status and retriever.last_filter is None


def test_scopes_come_from_authenticated_role_and_pii_is_masked():
    retriever = FakeRetriever()
    assistant = make_assistant(["12 ngày [S1]."], retriever=retriever)
    response = assistant.answer(QUESTION + " SĐT 0912345678", UserContext("canbo", "officer"), log=False)
    assert retriever.last_filter.scopes == ("public", "internal")
    assert response.warnings and "0912345678" not in str(response.to_dict())
    assistant.answer(QUESTION, UserContext(), log=False)
    assert retriever.last_filter.scopes == ("public",)


def test_legacy_invoke_interface():
    result = make_assistant(["12 ngày [S1]."]).invoke({"query": QUESTION})
    assert result["result"] == "12 ngày [S1]." and [d.metadata["chunk_id"] for d in result["source_documents"]] == [
        "c1", "c2"]
