import pytest

from evaluation.generation_metrics import bleu4, generation_scores, rouge_l, rouge_n
from evaluation.run_benchmark import contains_key_facts, matches_gold, retrieval_metrics
from src import telemetry
from src.assistant import AssistantResponse


def test_rouge_and_bleu():
    ref = "thời hạn giải quyết 03 ngày làm việc"
    assert rouge_n(ref, ref, 1) == 1.0 and rouge_l(ref, ref) == 1.0 and bleu4(ref, ref) == pytest.approx(1.0)
    assert rouge_n("hoàn toàn khác", ref, 1) == 0.0
    partial = generation_scores("giải quyết trong 03 ngày", ref)
    assert 0 < partial["rouge1"] < 1 and 0 < partial["rougeL"] < 1 and partial["bleu4"] < partial["rouge1"]


GOLD_TTHC = {"document_id": "tthc", "procedure_id": "p1", "field": "ho_so"}
GOLD_LAW = {"document_id": "luat", "article": "Điều 8"}


def card(document_id, procedure_id=None, field=None, location=""):
    return {"document_id": document_id, "procedure_id": procedure_id, "procedure_field": field, "location": location}


def test_gold_matching():
    assert matches_gold(card("tthc", "p1", "ho_so"), GOLD_TTHC)
    assert not matches_gold(card("tthc", "p1", "trinh_tu"), GOLD_TTHC)
    assert matches_gold(card("luat", location="Điều 8"), GOLD_LAW)
    assert not matches_gold(card("luat", location="Điều 80"), GOLD_LAW)


def test_retrieval_metrics_stay_in_range_when_a_section_has_several_parts():
    ranked = [card("tthc", "p1", "ho_so"), card("tthc", "p1", "ho_so"), card("x"), card("luat", location="Điều 8")]
    metrics = retrieval_metrics(ranked, [GOLD_TTHC, GOLD_LAW])
    assert metrics["recall@5"] == 1.0 and metrics["mrr@10"] == 1.0
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert 0 < metrics["ndcg@5"] <= 1.0


def test_key_facts():
    assert contains_key_facts("Nam đủ 20 tuổi, nữ đủ 18 tuổi", [["20 tuổi"], ["18 tuổi"]])
    assert not contains_key_facts("Nam đủ 20 tuổi", [["20 tuổi"], ["18 tuổi"]])


def test_request_log_does_not_store_question_text_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "LOG_QUESTION_TEXT", False)
    response = AssistantResponse("answered_with_sources", "a", "req-0123456789abcdef",
                                 sources=[{"chunk_id": "c1"}], related_sources=[{"chunk_id": "c2"}],
                                 timings={"total_ms": 1200.0})
    telemetry.log_request(response, "câu hỏi bí mật", "citizen", "web", directory=tmp_path)
    telemetry.log_request(AssistantResponse("insufficient_evidence", "b", "req-1", timings={"total_ms": 300.0}),
                          "q2", "citizen", "web", directory=tmp_path)
    entries = telemetry.read_log("requests.jsonl", directory=tmp_path)
    assert entries[0]["question"] is None and "bí mật" not in str(entries)
    assert entries[0]["sources"] == ["c1"] and entries[0]["retrieved"] == ["c1", "c2"]
    stats = telemetry.summary(directory=tmp_path)
    assert stats["requests"] == 2 and stats["insufficient_rate"] == 0.5 and stats["latency_ms"]["p95"] == 1200.0
