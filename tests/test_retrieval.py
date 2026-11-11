import pytest
from langchain_core.documents import Document

from src.knowledge.filters import SearchFilter
from src.retrieval_hybrid import (Candidate, HybridRetriever, limit_per_document, reciprocal_rank_fusion, tokenize,
                                  weighted_fusion)


def doc(chunk_id: str, text: str, **meta) -> Document:
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "access_scope": "public", **meta})


class FakeStore:
    """Dense store giả: trả về theo thứ tự cố định, có áp dụng bộ lọc như Qdrant."""

    def __init__(self, ranked: list[Document]):
        self.ranked = ranked

    def search(self, query, k, search_filter):
        allowed = [d for d in self.ranked if search_filter.allows(d.metadata)]
        return [(d, 1.0 - 0.01 * i) for i, d in enumerate(allowed[:k])]


def test_tokenize_keeps_document_numbers_and_diacritics():
    assert tokenize("Phạt theo Nghị định 168/2024/NĐ-CP, mức 1.000.000 đồng") == [
        "phạt", "theo", "nghị", "định", "168/2024/nđ-cp", "mức", "1.000.000", "đồng"]


def test_rrf_follows_formula():
    a, b, c = doc("a", "x"), doc("b", "y"), doc("c", "z")
    fused = dict((d.metadata["chunk_id"], s) for d, s in reciprocal_rank_fusion([[a, b], [b, c]], k0=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)    # hạng 2 ở danh sách 1, hạng 1 ở danh sách 2
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    assert max(fused, key=fused.get) == "b"


def test_weighted_fusion_is_kept_for_comparison():
    a, b = doc("a", "x"), doc("b", "y")
    fused = weighted_fusion([(a, 10.0), (b, 5.0)], [(b, 0.9), (a, 0.1)], alpha=0.5)
    assert {d.metadata["chunk_id"] for d, _ in fused} == {"a", "b"}


def test_limit_per_document_counts_each_procedure_separately():
    same_doc = [Candidate(doc(f"p{i}-{j}", f"text {i} {j}", document_id="tthc", procedure_id=f"p{i}"), 1.0)
                for i in range(3) for j in range(3)]
    kept = limit_per_document(same_doc, max_per_document=2)
    assert len(kept) == 6                                 # 2 đoạn × 3 thủ tục, không phải 2 cho cả bộ TTHC
    duplicate = limit_per_document([Candidate(doc("x", "trùng"), 1.0), Candidate(doc("y", "trùng"), 1.0)], 5)
    assert len(duplicate) == 1


def test_access_filter_applies_to_both_sparse_and_dense():
    public = doc("pub", "hướng dẫn tiếp nhận hồ sơ khai sinh")
    internal = doc("int", "hướng dẫn nội bộ tiếp nhận hồ sơ khai sinh mã NB-01", access_scope="internal")
    retriever = HybridRetriever([public, internal], FakeStore([internal, public]), use_cross_encoder=False)
    citizen = retriever.hybrid_search("hướng dẫn tiếp nhận hồ sơ", SearchFilter(scopes=("public",)))
    assert [c.doc.metadata["chunk_id"] for c in citizen] == ["pub"]
    assert retriever.retrieve_bm25("nội bộ NB-01", 5, SearchFilter(scopes=("public",))) == []
    officer = retriever.hybrid_search("hướng dẫn nội bộ", SearchFilter(scopes=("public", "internal")))
    assert {c.doc.metadata["chunk_id"] for c in officer} == {"pub", "int"}


def test_section_expansion_brings_the_asked_section_of_top_procedure():
    steps = doc("ks-tt", "[TTHC: Đăng ký khai sinh] Trình tự: nộp giấy tờ tại UBND", procedure_id="ks",
                procedure_field="trinh_tu", procedure_title="Thủ tục đăng ký khai sinh", document_id="tthc")
    dossier = doc("ks-hs", "[TTHC: Đăng ký khai sinh] Thành phần hồ sơ: tờ khai, giấy chứng sinh", procedure_id="ks",
                  procedure_field="ho_so", procedure_title="Thủ tục đăng ký khai sinh", document_id="tthc")
    retriever = HybridRetriever([steps, dossier], FakeStore([steps]), use_cross_encoder=False)
    results = retriever.hybrid_search("nộp giấy tờ", SearchFilter(), field_hint="ho_so", k_final=5)
    assert [c.doc.metadata["chunk_id"] for c in results] == ["ks-tt", "ks-hs"]
    assert results[1].scores["expanded"] == 1.0


def test_title_coverage_prefers_exact_procedure():
    query = set(tokenize("đăng ký kết hôn mất bao lâu"))
    exact = HybridRetriever._title_coverage(query, "Thủ tục đăng ký kết hôn")
    mobile = HybridRetriever._title_coverage(query, "Thủ tục đăng ký kết hôn lưu động")
    assert exact == 1.0 and mobile < exact
