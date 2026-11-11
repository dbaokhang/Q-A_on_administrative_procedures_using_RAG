"""
Chạy benchmark định dạng Vi-CommuneQA và tổng hợp chỉ số theo Bảng 1.7 / Bảng 2.6–2.7.

    python evaluation/run_benchmark.py                          # đầy đủ: truy xuất + sinh câu trả lời (gọi LLM)
    python evaluation/run_benchmark.py --retrieval-only         # chỉ truy xuất, không gọi LLM sinh câu trả lời
    python evaluation/run_benchmark.py --bertscore              # thêm BERTScore (cần: pip install bert-score)

Mỗi dòng của tệp benchmark (JSONL): id, question, role, intent, expected_status, gold (danh sách đoạn chuẩn theo
document_id + procedure_id/field hoặc article), reference_answer, must_include, forbid_scope (tùy chọn).

Chỉ số:
- Truy xuất: Recall@5, Precision@3, MRR@10, NDCG@5 so với gold (chỉ câu có gold).
- Sinh: ROUGE-1/2/L, BLEU-4 (+ BERTScore) so với reference_answer (chỉ câu trả lời có nguồn).
- Kiểm soát: độ chính xác trạng thái (trả lời / hỏi lại / từ chối / không đủ quyền), tỷ lệ trích dẫn đúng nguồn
  chuẩn, tỷ lệ chứa đủ ý chính (must_include), số lần lộ tài liệu ngoài quyền.
- Độ trễ: trung bình, P50, P95 từng giai đoạn và toàn bộ.
Kết quả ghi vào evaluation/results_benchmark_<ngày>.json kèm cấu hình pipeline để tái lập.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.generation_metrics import bertscore, generation_scores  # noqa: E402

DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "benchmark" / "vi_communeqa_sample.jsonl"
_ARTICLE = re.compile(r"Điều\s+\d+[a-zđ]?")


def load_benchmark(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def matches_gold(card: dict, gold: dict) -> bool:
    """Một nguồn khớp đoạn chuẩn khi cùng tài liệu và (nếu nhãn có) cùng thủ tục + mục, hoặc cùng số Điều."""
    if card.get("document_id") != gold.get("document_id"):
        return False
    if gold.get("procedure_id"):
        if card.get("procedure_id") != gold["procedure_id"]:
            return False
        return not gold.get("field") or card.get("procedure_field") == gold["field"]
    if gold.get("article"):
        found = _ARTICLE.search(card.get("location") or "")
        return bool(found) and found.group(0) == gold["article"]
    return True


def retrieval_metrics(ranked: list[dict], gold: list[dict]) -> dict[str, float]:
    relevant = [any(matches_gold(card, g) for g in gold) for card in ranked]
    found_at_5 = sum(any(matches_gold(card, g) for card in ranked[:5]) for g in gold)
    first = next((i for i, rel in enumerate(relevant[:10], 1) if rel), None)
    # NDCG nhị phân theo từng đoạn chuẩn: một mục bị tách nhiều phần (phần 1/3, 2/3) chỉ được tính một lần
    credited, gains = set(), []
    for card in ranked[:5]:
        hit = next((j for j, g in enumerate(gold) if j not in credited and matches_gold(card, g)), None)
        gains.append(hit is not None)
        if hit is not None:
            credited.add(hit)
    dcg = sum(1 / math.log2(i + 1) for i, gain in enumerate(gains, 1) if gain)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), 5) + 1))
    return {
        "recall@5": found_at_5 / len(gold),
        "precision@3": sum(relevant[:3]) / 3,
        "mrr@10": 1 / first if first else 0.0,
        "ndcg@5": dcg / ideal if ideal else 0.0,
    }


def contains_key_facts(answer: str, must_include: list[list[str]]) -> bool:
    lowered = answer.lower()
    return all(any(option.lower() in lowered for option in group) for group in must_include)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def run(args) -> dict:
    from src.assistant import build_assistant, source_card
    from src.auth import UserContext

    assistant = build_assistant()
    cases = load_benchmark(Path(args.benchmark))
    if args.limit:
        cases = cases[:args.limit]
    rows = []
    for i, case in enumerate(cases, 1):
        user = UserContext(f"benchmark-{case['role']}", case["role"])
        print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:70]}")
        started = time.perf_counter()
        if args.retrieval_only:
            timings: dict = {}
            _, intent, candidates = assistant.retrieve(case["question"], user, timings=timings)
            ranked = [source_card(c.doc, f"S{n}", c.score) for n, c in enumerate(candidates, 1)]
            row = {"status": None, "answer": "", "intent": intent.name, "ranked": ranked, "cited": [],
                   "timings": {**timings, "total_ms": (time.perf_counter() - started) * 1000}}
        else:
            response = assistant.answer(case["question"], user, channel="benchmark", log=False)
            ranked = sorted(response.sources + response.related_sources, key=lambda s: int(s["label"][1:]))
            row = {"status": response.status, "answer": response.answer, "intent": response.intent,
                   "ranked": ranked, "cited": response.sources, "timings": response.timings}
        row["case"] = case
        rows.append(row)
        if args.delay:
            time.sleep(args.delay)
    return summarize(rows, assistant.pipeline, args)


def summarize(rows: list[dict], pipeline: dict, args) -> dict:
    retrieval, generation, per_case = [], [], []
    status_ok = cited_gold = key_facts = answered_gold = leaks = intent_ok = intent_n = 0
    hypotheses, references = [], []
    for row in rows:
        case = row["case"]
        entry = {"id": case["id"], "status": row["status"], "expected_status": case["expected_status"],
                 "intent": row["intent"], "expected_intent": case.get("intent"),
                 "retrieved": [f"{c['title'][:40]} | {c['location'][:50]}" for c in row["ranked"]],
                 "total_ms": round(row["timings"].get("total_ms", 0), 1)}
        if row["intent"]:
            intent_ok += row["intent"] == case.get("intent")
            intent_n += 1
        if case.get("forbid_scope"):
            leaked = [c for c in row["ranked"] if c.get("access_scope") == case["forbid_scope"]]
            leaks += bool(leaked)
            entry["leaked"] = len(leaked)
        if case["gold"] and row["ranked"] is not None:
            metrics = retrieval_metrics(row["ranked"], case["gold"])
            retrieval.append(metrics)
            entry["retrieval"] = metrics
        if row["status"] is not None:
            status_ok += row["status"] == case["expected_status"]
            if case["gold"] and case["expected_status"] == "answered_with_sources":
                answered_gold += 1
                cited_gold += any(matches_gold(c, g) for c in row["cited"] for g in case["gold"])
                key_facts += row["status"] == "answered_with_sources" and contains_key_facts(
                    row["answer"], case.get("must_include", []))
                if row["status"] == "answered_with_sources" and case.get("reference_answer"):
                    scores = generation_scores(row["answer"], case["reference_answer"])
                    generation.append(scores)
                    entry["generation"] = scores
                    hypotheses.append(row["answer"])
                    references.append(case["reference_answer"])
            entry["answer"] = row["answer"]
        per_case.append(entry)

    stages = sorted({k for row in rows for k in row["timings"] if k.endswith("_ms")})
    latency = {stage: {
        "mean": _mean([r["timings"][stage] for r in rows if stage in r["timings"]]),
        "p50": _percentile([r["timings"][stage] for r in rows if stage in r["timings"]], 0.5),
        "p95": _percentile([r["timings"][stage] for r in rows if stage in r["timings"]], 0.95),
    } for stage in stages}
    result = {
        "date": date.today().isoformat(),
        "benchmark": str(Path(args.benchmark).relative_to(PROJECT_ROOT)),
        "mode": "retrieval-only" if args.retrieval_only else "full",
        "pipeline": pipeline,
        "n_cases": len(rows),
        "retrieval": {k: _mean([m[k] for m in retrieval]) for k in ("recall@5", "precision@3", "mrr@10", "ndcg@5")},
        "retrieval_n": len(retrieval),
        "intent_accuracy": round(intent_ok / intent_n, 4) if intent_n else None,
        "intent_n": intent_n,
        "access_leaks": leaks,
        "latency_ms": latency,
        "cases": per_case,
    }
    if not args.retrieval_only:
        result["control"] = {
            "status_accuracy": round(status_ok / len(rows), 4) if rows else None,
            "citation_gold_rate": round(cited_gold / answered_gold, 4) if answered_gold else None,
            "key_facts_rate": round(key_facts / answered_gold, 4) if answered_gold else None,
            "answerable_n": answered_gold,
        }
        result["generation"] = {k: _mean([g[k] for g in generation]) for k in ("rouge1", "rouge2", "rougeL", "bleu4")}
        result["generation_n"] = len(generation)
        if args.bertscore and hypotheses:
            scores = bertscore(hypotheses, references)
            result["generation"]["bertscore_f1"] = _mean(scores) if scores else "bert-score chưa được cài"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.0, help="nghỉ giữa các câu (giới hạn API)")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = run(args)
    suffix = "_retrieval" if args.retrieval_only else ""
    out = Path(args.out or PROJECT_ROOT / "evaluation" / f"results_benchmark{suffix}_{result['date']}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k not in ("cases", "pipeline")}, ensure_ascii=False, indent=2))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
