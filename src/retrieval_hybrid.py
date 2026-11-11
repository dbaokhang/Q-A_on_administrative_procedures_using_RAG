"""
Truy xuất lai (mục 2.2.2 b–f): BM25 (sparse) + vector ngữ nghĩa (dense) → hợp nhất RRF → khử trùng lặp,
giới hạn số đoạn mỗi tài liệu → tái xếp hạng Cross-Encoder (tùy chọn).

Lọc quyền / hiệu lực (SearchFilter) áp dụng ở CẢ HAI nhánh trước khi hợp nhất, nên đoạn ngoài quyền không bao
giờ đi vào danh sách ứng viên hay ngữ cảnh mô hình (BR-04).
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .knowledge.filters import SearchFilter
except ImportError:
    from knowledge.filters import SearchFilter


class DocumentLike(Protocol):
    """Minimal document contract used by the retriever."""

    page_content: str
    metadata: Dict[str, Any]


class DenseStore(Protocol):
    def search(self, query: str, k: int, search_filter: SearchFilter) -> List[Tuple[DocumentLike, float]]: ...


# Import cross-encoder for re-ranking
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("Warning: CrossEncoder not available. Install with: pip install sentence-transformers")

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# FUSION_METHOD: "rrf" (mặc định, mục 2.2.2 d) | "weighted" (cộng điểm đã chuẩn hóa — cấu hình cũ, để so sánh)
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf").strip().lower()
RRF_K = int(os.getenv("RRF_K", "60"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").strip().lower() != "false"
# Số ứng viên đưa vào Cross-Encoder: chi phí rerank tăng tuyến tính theo số này (Bảng 2.9: đánh đổi độ trễ)
RERANK_POOL = int(os.getenv("RERANK_POOL", "24"))

_TOKEN = re.compile(r"[^\W_]+(?:[./-][^\W_]+)*", re.UNICODE)
_TITLE_STOPWORDS = {"thủ", "tục", "các", "về", "cho", "trong", "của", "hành", "chính", "và", "việc"}


def tokenize(text: str) -> List[str]:
    """Tách từ cho BM25: chữ thường, giữ nguyên dấu tiếng Việt và số hiệu ("168/2024/nđ-cp", "1.000.000")."""
    return _TOKEN.findall(text.lower())


@dataclass
class Candidate:
    doc: DocumentLike
    score: float                       # điểm cuối dùng để xếp hạng (rerank nếu có, ngược lại điểm hợp nhất)
    scores: Dict[str, float] = field(default_factory=dict)   # bm25_rank, dense_rank, rrf, rerank... để gỡ lỗi


def document_key(doc: DocumentLike) -> str:
    """Định danh ổn định của một đoạn; chunk cũ (không có chunk_id) dùng nguồn + nội dung."""
    if doc.metadata.get("chunk_id"):
        return doc.metadata["chunk_id"]
    return f"{doc.metadata.get('source', '')}:{doc.metadata.get('start_index', -1)}:{hash(doc.page_content)}"


def reciprocal_rank_fusion(ranked_lists: List[List[DocumentLike]], k0: int = RRF_K) -> List[Tuple[DocumentLike, float]]:
    """RRF(d) = Σ_i 1 / (k0 + r_i(d)); r bắt đầu từ 1. Đoạn không có trong một danh sách không được cộng điểm."""
    scores: Dict[str, float] = {}
    docs: Dict[str, DocumentLike] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            key = document_key(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k0 + rank)
    return sorted(((docs[key], score) for key, score in scores.items()), key=lambda item: item[1], reverse=True)


def normalize_scores(results: List[Tuple[DocumentLike, float]]) -> List[Tuple[DocumentLike, float]]:
    if not results:
        return []
    raw = [score for _, score in results]
    low, high = min(raw), max(raw)
    if high == low:
        return [(doc, 1.0) for doc, _ in results]
    return [(doc, (score - low) / (high - low)) for doc, score in results]


def weighted_fusion(bm25: List[Tuple[DocumentLike, float]], dense: List[Tuple[DocumentLike, float]],
                    alpha: float = 0.5) -> List[Tuple[DocumentLike, float]]:
    """Cấu hình cũ: alpha × dense + (1 − alpha) × BM25 sau khi chuẩn hóa min-max từng nhánh."""
    combined: Dict[str, Dict[str, Any]] = {}
    for doc, score in normalize_scores(bm25):
        combined[document_key(doc)] = {"doc": doc, "bm25": score, "dense": 0.0}
    for doc, score in normalize_scores(dense):
        entry = combined.setdefault(document_key(doc), {"doc": doc, "bm25": 0.0, "dense": 0.0})
        entry["dense"] = score
    fused = [(e["doc"], alpha * e["dense"] + (1 - alpha) * e["bm25"]) for e in combined.values()]
    return sorted(fused, key=lambda item: item[1], reverse=True)


def unit_key(doc: DocumentLike) -> str:
    """Đơn vị giới hạn số đoạn: từng thủ tục với bộ TTHC (cả bộ chung một document_id), từng văn bản với VBQPPL."""
    meta = doc.metadata
    return meta.get("procedure_id") or meta.get("document_id") or meta.get("source", "")


def limit_per_document(candidates: List[Candidate], max_per_document: int) -> List[Candidate]:
    """Khử trùng lặp nội dung và giới hạn số đoạn từ cùng một văn bản (tránh một văn bản chiếm hết ngữ cảnh)."""
    seen_text, per_doc, kept = set(), {}, []
    for cand in candidates:
        text_key = hash(cand.doc.page_content)
        doc_id = unit_key(cand.doc)
        if text_key in seen_text or per_doc.get(doc_id, 0) >= max_per_document:
            continue
        seen_text.add(text_key)
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        kept.append(cand)
    return kept


class HybridRetriever:
    """BM25 + dense + RRF + Cross-Encoder trên một tập chunk và một DenseStore cùng tập đó."""

    def __init__(self, documents: List[DocumentLike], store: DenseStore, use_cross_encoder: bool = USE_RERANKER,
                 cross_encoder=None, cross_encoder_model: str = CROSS_ENCODER_MODEL,
                 fusion: str = FUSION_METHOD, alpha: float = 0.5):
        self.documents = documents
        self.store = store
        self.fusion = fusion
        self.alpha = alpha
        print("Initializing BM25 retriever...")
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in documents])
        # (mã thủ tục, mục) → các đoạn: để bổ sung đúng mục được hỏi của thủ tục đang đứng đầu
        self.procedure_sections: Dict[Tuple[str, str], List[DocumentLike]] = {}
        for doc in documents:
            if doc.metadata.get("procedure_id"):
                key = (doc.metadata["procedure_id"], doc.metadata.get("procedure_field"))
                self.procedure_sections.setdefault(key, []).append(doc)
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        elif self.use_cross_encoder:
            print(f"Loading cross-encoder: {cross_encoder_model}")
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model, device="cpu")
            except Exception as exc:
                print(f"Warning: could not load cross-encoder, falling back to BM25 + dense only: {exc}")
                self.cross_encoder, self.use_cross_encoder = None, False
        else:
            self.cross_encoder = None

    def retrieve_bm25(self, query: str, k: int, search_filter: SearchFilter) -> List[Tuple[DocumentLike, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        allowed = np.array([search_filter.allows(doc.metadata) for doc in self.documents], dtype=bool)
        scores = np.where(allowed, scores, 0.0)
        top = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in top if scores[i] > 0]

    def retrieve_dense(self, query: str, k: int, search_filter: SearchFilter) -> List[Tuple[DocumentLike, float]]:
        return self.store.search(query, k, search_filter)

    def rerank(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        if not self.cross_encoder or not candidates:
            return candidates
        scores = self.cross_encoder.predict([(query, cand.doc.page_content) for cand in candidates])
        for cand, score in zip(candidates, scores):
            cand.scores["rerank"] = float(score)
            cand.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _expand_sections(self, candidates: List[Candidate], field: str, search_filter: SearchFilter,
                         top_n: int = 10) -> List[Candidate]:
        """Đưa mục `field` của các thủ tục đang đứng đầu lên ngay sau top_n (mở rộng sang đoạn cùng thủ tục,
        mục 2.3.1 d) — kể cả đoạn đã có trong danh sách nhưng ở thứ hạng quá thấp để vào nhóm tái xếp hạng."""
        existing = {document_key(c.doc): c for c in candidates}
        procedures = []
        for cand in candidates[:top_n]:
            pid = cand.doc.metadata.get("procedure_id")
            if pid and pid not in procedures:
                procedures.append(pid)
        head = candidates[:top_n]
        head_keys = {document_key(c.doc) for c in head}
        moved = []
        for pid in procedures:
            for doc in self.procedure_sections.get((pid, field), []):
                key = document_key(doc)
                if key in head_keys or not search_filter.allows(doc.metadata):
                    continue
                cand = existing.get(key) or Candidate(doc, 0.0, {"fusion": 0.0})
                cand.scores["expanded"] = 1.0
                head_keys.add(key)
                moved.append(cand)
        moved_keys = {document_key(c.doc) for c in moved}
        rest = [c for c in candidates[top_n:] if document_key(c.doc) not in moved_keys]
        return head + moved + rest

    @staticmethod
    def _title_coverage(query_tokens: set, title: str) -> float:
        """Tỷ lệ từ trong tên thủ tục có trong truy vấn: "đăng ký kết hôn" khớp đủ thủ tục "Đăng ký kết hôn",
        chỉ khớp một phần "Đăng ký kết hôn lưu động" → ưu tiên thủ tục đúng như người hỏi nêu (FR-07)."""
        tokens = set(tokenize(title)) - _TITLE_STOPWORDS
        return len(tokens & query_tokens) / len(tokens) if tokens else 0.0

    def hybrid_search(self, query: str, search_filter: SearchFilter, k_sparse: int = 20, k_dense: int = 20,
                      k_final: int = 7, max_extra: int = 3, score_margin: float = 2.0, max_per_document: int = 5,
                      rerank: bool = True, timings: Dict[str, float] | None = None,
                      field_hint: str | None = None, field_bonus: float = 2.0,
                      title_bonus: float = 1.5) -> List[Candidate]:
        """Trả về tối đa k_final (+ max_extra đoạn sát điểm đoạn đầu khi có rerank) ứng viên đã xếp hạng.

        k thích ứng: câu hỏi nhiều đối tượng (vd. vượt đèn đỏ: ô tô, xe máy, xe đạp — mỗi đối tượng một Điều)
        cần hơn k đoạn; đoạn ngoài top-k chỉ được thêm khi điểm cross-encoder sát đoạn đứng đầu.
        field_hint (từ phân loại ý định): mục TTHC được hỏi → bổ sung mục đó của thủ tục đứng đầu và cộng
        field_bonus vào điểm rerank của các đoạn thuộc mục đó.
        """
        timings = timings if timings is not None else {}
        t0 = time.perf_counter()
        sparse = self.retrieve_bm25(query, k_sparse, search_filter)
        t1 = time.perf_counter()
        dense = self.retrieve_dense(query, k_dense, search_filter)
        t2 = time.perf_counter()
        timings.update(sparse_ms=(t1 - t0) * 1000, dense_ms=(t2 - t1) * 1000)

        sparse_rank = {document_key(doc): r for r, (doc, _) in enumerate(sparse, 1)}
        dense_rank = {document_key(doc): r for r, (doc, _) in enumerate(dense, 1)}
        if self.fusion == "weighted":
            fused = weighted_fusion(sparse, dense, self.alpha)
        else:
            fused = reciprocal_rank_fusion([[d for d, _ in sparse], [d for d, _ in dense]])
        candidates = [
            Candidate(doc, score, {"fusion": score,
                                   "bm25_rank": sparse_rank.get(document_key(doc)),
                                   "dense_rank": dense_rank.get(document_key(doc))})
            for doc, score in fused
        ]
        candidates = limit_per_document(candidates, max_per_document)
        if field_hint:
            candidates = self._expand_sections(candidates, field_hint, search_filter)
        t3 = time.perf_counter()
        timings["fusion_ms"] = (t3 - t2) * 1000

        if rerank and self.use_cross_encoder:
            pool = candidates[:max(k_final + max_extra, RERANK_POOL)]
            ranked = self.rerank(query, pool)
            query_tokens = set(tokenize(query))
            for cand in ranked:
                meta = cand.doc.metadata
                if field_hint and meta.get("procedure_field") == field_hint:
                    cand.scores["field_bonus"] = field_bonus
                    cand.score += field_bonus
                if meta.get("procedure_title"):
                    bonus = title_bonus * self._title_coverage(query_tokens, meta["procedure_title"])
                    cand.scores["title_bonus"] = round(bonus, 3)
                    cand.score += bonus
            ranked.sort(key=lambda c: c.score, reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t3) * 1000
            if not ranked:
                return []
            top = ranked[0].score
            extra = [c for c in ranked[k_final:k_final + max_extra] if c.score >= top - score_margin]
            return ranked[:k_final] + extra
        return candidates[:k_final]
