"""
Chỉ số sinh câu trả lời so với câu trả lời tham chiếu (mục 1.4.2): ROUGE-1/2/L, BLEU-4, BERTScore.

Tách từ theo âm tiết (chữ thường, bỏ dấu câu) — đơn vị tự nhiên của tiếng Việt khi không dùng bộ tách từ.
BERTScore cần gói bert-score (pip install bert-score) và tải mô hình đa ngôn ngữ; không có thì bỏ qua.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[^\W_]+(?:[.,/][0-9]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, hyp_total: int, ref_total: int) -> float:
    if not overlap or not hyp_total or not ref_total:
        return 0.0
    precision, recall = overlap / hyp_total, overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    hyp, ref = _ngrams(tokenize(hypothesis), n), _ngrams(tokenize(reference), n)
    return _f1(sum((hyp & ref).values()), sum(hyp.values()), sum(ref.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    return _f1(_lcs_length(hyp, ref), len(hyp), len(ref))


def bleu4(hypothesis: str, reference: str) -> float:
    """BLEU-4 mức câu, trọng số đều 1/4, phạt độ ngắn (BP), làm mượt cộng-epsilon cho n-gram không khớp."""
    hyp, ref = tokenize(hypothesis), tokenize(reference)
    if not hyp or not ref:
        return 0.0
    log_precision = 0.0
    for n in range(1, 5):
        hyp_ngrams, ref_ngrams = _ngrams(hyp, n), _ngrams(ref, n)
        total = max(sum(hyp_ngrams.values()), 1)
        matches = sum((hyp_ngrams & ref_ngrams).values())
        log_precision += 0.25 * math.log((matches or 0.1) / total)
    brevity = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(log_precision)


def bertscore(hypotheses: list[str], references: list[str],
              model_type: str = "bert-base-multilingual-cased") -> list[float] | None:
    """F1 BERTScore từng câu; None nếu chưa cài bert-score (không chặn phần đánh giá còn lại)."""
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f1 = score(hypotheses, references, model_type=model_type, lang="vi", verbose=False)
    return [float(x) for x in f1]


def generation_scores(hypothesis: str, reference: str) -> dict[str, float]:
    return {
        "rouge1": rouge_n(hypothesis, reference, 1),
        "rouge2": rouge_n(hypothesis, reference, 2),
        "rougeL": rouge_l(hypothesis, reference),
        "bleu4": bleu4(hypothesis, reference),
    }
