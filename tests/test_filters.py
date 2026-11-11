"""Ma trận vai trò – tài liệu (mục 2.3.2 h, NFR-06) và lọc hiệu lực: cùng kết quả ở bộ lọc Python (BM25, FAISS)
và bộ lọc Qdrant (dense)."""

import datetime as dt

import pytest

from src.auth import ROLES
from src.knowledge.filters import SearchFilter
from src.knowledge.schema import date_ordinal

DOCS = {
    "public": {"access_scope": "public"},
    "internal": {"access_scope": "internal"},
    "future": {"access_scope": "public", "effective_from_ord": date_ordinal("2030-01-01")},
    "expired": {"access_scope": "public", "effective_to_ord": date_ordinal("2020-01-01")},
    "superseded_2025": {"access_scope": "public", "superseded": True, "superseded_from_ord": date_ordinal("2025-01-01")},
    "superseded_unknown": {"access_scope": "public", "superseded": True, "superseded_from_ord": None},
}
TODAY = dt.date(2026, 11, 5)


def allowed(search_filter: SearchFilter) -> set[str]:
    return {name for name, meta in DOCS.items() if search_filter.allows(meta)}


@pytest.mark.parametrize("role, expected", [
    ("citizen", {"public"}),
    ("officer", {"public", "internal"}),
    ("admin", {"public", "internal"}),
])
def test_role_document_matrix(role, expected):
    flt = SearchFilter(scopes=ROLES[role]["scopes"], as_of=TODAY)
    assert allowed(flt) & {"public", "internal"} == expected


def test_effective_dates_and_superseded_parts():
    assert allowed(SearchFilter(as_of=TODAY)) == {"public"}
    # Hỏi về thời điểm trước khi bị thay thế → phần đó vẫn được tìm (BR-10)
    assert "superseded_2025" in allowed(SearchFilter(as_of=dt.date(2024, 6, 1)))
    assert "expired" in allowed(SearchFilter(as_of=dt.date(2019, 6, 1)))
    # Người hỏi nhắc thẳng văn bản cũ
    assert {"superseded_2025", "superseded_unknown"} <= allowed(SearchFilter(as_of=TODAY, include_superseded=True))


@pytest.mark.parametrize("search_filter", [
    SearchFilter(scopes=("public",), as_of=TODAY),
    SearchFilter(scopes=("public", "internal"), as_of=TODAY),
    SearchFilter(scopes=("public",), as_of=dt.date(2024, 6, 1)),
    SearchFilter(scopes=("public",), as_of=dt.date(2019, 6, 1)),
    SearchFilter(scopes=("public",), as_of=TODAY, include_superseded=True),
])
def test_qdrant_filter_matches_python_filter(search_filter):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    names = list(DOCS)
    client.upsert("t", points=[
        models.PointStruct(id=i, vector=[1.0, 0.1 * i], payload={**DOCS[name], "name": name})
        for i, name in enumerate(names)
    ])
    hits = client.query_points("t", query=[1.0, 0.0], query_filter=search_filter.to_qdrant(), limit=20,
                               with_payload=True).points
    assert {hit.payload["name"] for hit in hits} == allowed(search_filter)
