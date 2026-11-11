"""
Kho vector cho dense retrieval.

- Qdrant (mặc định, VECTOR_BACKEND=qdrant): chế độ nhúng cục bộ (qdrant_data/, không cần Docker) hoặc server
  (QDRANT_URL). Payload chứa metadata → lọc quyền / hiệu lực ngay trong truy vấn vector.
- FAISS (VECTOR_BACKEND=faiss): phương án so sánh; metadata lọc ngoài chỉ mục (SearchFilter.allows).

Đánh chỉ mục an toàn: mỗi phiên bản kho dựng vào một collection mới, kiểm tra số bản ghi rồi mới chuyển
manifest sang collection đó; vector của nội dung không đổi được dùng lại (không embed lại), nên nạp lại là
idempotent và việc thêm một tài liệu chỉ phải embed phần mới.

Lưu ý: Qdrant cục bộ khóa thư mục — mỗi lúc chỉ một tiến trình (Streamlit HOẶC API HOẶC đánh giá) mở được.
Chạy đồng thời nhiều tiến trình thì dùng Qdrant server (QDRANT_URL).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_core.documents import Document

try:
    from .knowledge.filters import SearchFilter
    from .knowledge.schema import now_iso
except ImportError:
    from knowledge.filters import SearchFilter
    from knowledge.schema import now_iso

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", PROJECT_ROOT / "qdrant_data"))
LEGACY_FAISS_PATH = PROJECT_ROOT / "faiss_index_local"
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant").strip().lower()
EMBED_BATCH = 32


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _to_document(payload: dict) -> Document:
    payload = dict(payload)
    text = payload.pop("page_content")
    payload.pop("content_hash", None)
    return Document(page_content=text, metadata=payload)


def _embed_missing(embeddings, chunks: list[Document], cache: dict[str, list[float]],
                   log: Callable[[str], None]) -> list[list[float]]:
    missing = [c for c in chunks if content_hash(c.page_content) not in cache]
    if missing:
        log(f"Embedding {len(missing)}/{len(chunks)} đoạn mới...")
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        for chunk, vector in zip(batch, embeddings.embed_documents([c.page_content for c in batch])):
            cache[content_hash(chunk.page_content)] = list(vector)
        if start and start % (EMBED_BATCH * 10) == 0:
            log(f"  ... {start}/{len(missing)}")
    return [cache[content_hash(c.page_content)] for c in chunks]


def legacy_faiss_vectors(embeddings, log: Callable[[str], None] = print) -> dict[str, list[float]]:
    """Vector trong faiss_index_local/ (dựng sẵn bằng fine-tuned E5) → dùng lại nếu đúng cùng mô hình.

    Kiểm tra bằng cách embed lại một đoạn và so cosine: khác mô hình thì bỏ qua, không trộn không gian vector.
    """
    if not (LEGACY_FAISS_PATH / "index.faiss").exists():
        return {}
    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.load_local(str(LEGACY_FAISS_PATH), embeddings, allow_dangerous_deserialization=True)
        first_id = store.index_to_docstore_id[0]
        probe = store.docstore.search(first_id).page_content
        expected = np.asarray(embeddings.embed_query(probe), dtype=np.float32)
        stored = store.index.reconstruct(0)
        if len(stored) != len(expected):
            return {}
        cosine = float(np.dot(stored, expected) / (np.linalg.norm(stored) * np.linalg.norm(expected)))
        if cosine < 0.99:
            log(f"FAISS cũ không cùng mô hình embedding (cosine {cosine:.3f}) — không dùng lại.")
            return {}
        vectors = {}
        for position, doc_id in store.index_to_docstore_id.items():
            vectors[content_hash(store.docstore.search(doc_id).page_content)] = store.index.reconstruct(position).tolist()
        log(f"Dùng lại {len(vectors)} vector từ faiss_index_local/.")
        return vectors
    except Exception as e:  # chỉ là tối ưu: lỗi thì embed lại bình thường
        log(f"Không đọc được faiss_index_local/: {e}")
        return {}


class QdrantKnowledgeStore:
    backend = "qdrant"

    def __init__(self, embeddings, model_key: str, path: Path = QDRANT_PATH, url: str | None = None,
                 prefix: str = "knowledge"):
        from qdrant_client import QdrantClient

        self.embeddings = embeddings
        self.model_key = model_key
        self.base_name = f"{prefix}_{model_key}".replace("-", "_")
        self.client = QdrantClient(url=url) if url else QdrantClient(path=str(path))
        self.manifest_path = (Path(path) if not url else PROJECT_ROOT / "data") / f"manifest_{self.base_name}.json"

    # ---- manifest ----
    def manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    @property
    def collection(self) -> str | None:
        return self.manifest().get("collection")

    def _existing_vectors(self) -> dict[str, list[float]]:
        collection = self.collection
        if not collection or not self.client.collection_exists(collection):
            return {}
        cache, offset = {}, None
        while True:
            points, offset = self.client.scroll(collection, limit=512, offset=offset,
                                                with_payload=["content_hash"], with_vectors=True)
            for point in points:
                cache[point.payload["content_hash"]] = point.vector
            if offset is None:
                return cache

    # ---- đánh chỉ mục ----
    def sync(self, chunks: list[Document], corpus_version: str, log: Callable[[str], None] = print) -> dict:
        from qdrant_client import models

        manifest = self.manifest()
        if manifest.get("corpus_version") == corpus_version and self.client.collection_exists(manifest["collection"]):
            return manifest
        started = time.time()
        cache = self._existing_vectors() or (
            legacy_faiss_vectors(self.embeddings, log) if self.model_key == "finetuned-e5" else {})
        vectors = _embed_missing(self.embeddings, chunks, cache, log)

        new_collection = f"{self.base_name}__{corpus_version}"
        if self.client.collection_exists(new_collection):
            self.client.delete_collection(new_collection)
        self.client.create_collection(
            new_collection, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
        points = [
            models.PointStruct(
                id=chunk.metadata["chunk_id"], vector=vector,
                payload={**chunk.metadata, "page_content": chunk.page_content,
                         "content_hash": content_hash(chunk.page_content)},
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(new_collection, points=points[start:start + 256])
        count = self.client.count(new_collection, exact=True).count
        if count != len(chunks):  # chưa chuyển sang bản mới nếu ghi thiếu
            raise RuntimeError(f"Ghi chỉ mục thiếu: {count}/{len(chunks)} điểm")

        old_collection = manifest.get("collection")
        manifest = {
            "collection": new_collection, "corpus_version": corpus_version, "embedding_model": self.model_key,
            "chunk_count": count, "dimension": len(vectors[0]), "built_at": now_iso(),
            "build_seconds": round(time.time() - started, 1),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if old_collection and old_collection != new_collection and self.client.collection_exists(old_collection):
            self.client.delete_collection(old_collection)
        log(f"Đã đánh chỉ mục {count} đoạn vào {new_collection} ({manifest['build_seconds']} s).")
        return manifest

    def load_chunks(self) -> list[Document]:
        collection = self.collection
        if not collection or not self.client.collection_exists(collection):
            return []
        chunks, offset = [], None
        while True:
            points, offset = self.client.scroll(collection, limit=512, offset=offset, with_payload=True)
            chunks.extend(_to_document(point.payload) for point in points)
            if offset is None:
                break
        chunks.sort(key=lambda c: (c.metadata.get("document_id", ""), c.metadata.get("chunk_index", 0)))
        return chunks

    # ---- truy vấn ----
    def search(self, query: str, k: int, search_filter: SearchFilter) -> list[tuple[Document, float]]:
        """Điểm cosine (cao hơn = gần hơn), đã lọc quyền / hiệu lực trong truy vấn."""
        vector = self.embeddings.embed_query(query)
        response = self.client.query_points(self.collection, query=vector, query_filter=search_filter.to_qdrant(),
                                            limit=k, with_payload=True)
        return [(_to_document(point.payload), float(point.score)) for point in response.points]

    def close(self) -> None:
        self.client.close()


class FaissKnowledgeStore:
    """Phương án so sánh: FAISS trong bộ nhớ, dựng từ cùng vector với Qdrant; lọc metadata ngoài chỉ mục."""

    backend = "faiss"

    def __init__(self, embeddings, chunks: list[Document], vectors: list[list[float]]):
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy

        self.embeddings = embeddings
        self.store = FAISS.from_embeddings(
            [(c.page_content, v) for c, v in zip(chunks, vectors)], embeddings,
            metadatas=[c.metadata for c in chunks], distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    def search(self, query: str, k: int, search_filter: SearchFilter) -> list[tuple[Document, float]]:
        # Vector đã chuẩn hóa → inner product = cosine; fetch_k rộng hơn vì lọc sau khi tìm
        return self.store.similarity_search_with_score(query, k=k, filter=search_filter.allows, fetch_k=k * 8)


def open_knowledge_store(embeddings, model_key: str, corpus_version: str,
                         build_chunks: Callable[[Callable[[], list[Document]]], list[Document]],
                         log: Callable[[str], None] = print):
    """Mở kho vector đúng phiên bản kho tri thức; khác phiên bản thì dựng lại (chỉ embed đoạn mới).

    build_chunks nhận hàm đọc chunk của chỉ mục hiện tại (để dùng lại khi tệp nguồn bị thiếu).
    Trả về (store, chunks, rebuilt). Kho không đổi → đọc chunk thẳng từ payload Qdrant, không trích xuất lại tệp.
    """
    qdrant = QdrantKnowledgeStore(embeddings, model_key, url=os.getenv("QDRANT_URL") or None)
    manifest = qdrant.manifest()
    rebuilt = not (manifest.get("corpus_version") == corpus_version and manifest.get("collection")
                   and qdrant.client.collection_exists(manifest["collection"]))
    if rebuilt:
        chunks = build_chunks(qdrant.load_chunks)
        qdrant.sync(chunks, corpus_version, log)
    chunks = qdrant.load_chunks()
    if VECTOR_BACKEND == "faiss":
        vectors = _embed_missing(embeddings, chunks, qdrant._existing_vectors(), log)
        qdrant.close()
        return FaissKnowledgeStore(embeddings, chunks, vectors), chunks, rebuilt
    return qdrant, chunks, rebuilt
