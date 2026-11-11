import pytest

from src.knowledge.corpus import corpus_version
from src.knowledge.registry import DocumentRegistry, RegistryError
from src.knowledge.schema import DocumentRecord


@pytest.fixture
def registry(tmp_path):
    return DocumentRegistry(tmp_path / "registry")


def make_record(tmp_path, doc_id: str, content: str = "nội dung", **kwargs) -> DocumentRecord:
    path = tmp_path / f"{doc_id}.txt"
    path.write_text(content, encoding="utf-8")
    return DocumentRecord(document_id=doc_id, title=doc_id, document_type="law", file_path=str(path),
                          status=kwargs.pop("status", "needs_review"), **kwargs)


def test_register_hashes_file_and_rejects_duplicates(registry, tmp_path):
    record = registry.register(make_record(tmp_path, "a"), "admin")
    assert len(record.file_hash) == 64 and record.version_id.startswith("a@")
    with pytest.raises(RegistryError, match="đã tồn tại"):
        registry.register(make_record(tmp_path, "a", content="khác"), "admin")
    with pytest.raises(RegistryError, match="trùng"):
        registry.register(make_record(tmp_path, "b"), "admin")  # cùng nội dung tệp với "a"


def test_only_approved_documents_are_served_and_transitions_are_checked(registry, tmp_path):
    registry.register(make_record(tmp_path, "a"), "admin")
    assert registry.servable() == []                         # BR-01: chờ kiểm tra thì chưa dùng để trả lời
    with pytest.raises(RegistryError, match="Không thể chuyển"):
        registry.set_status("a", "indexed", "admin")          # phải duyệt trước khi đánh chỉ mục
    registry.set_status("a", "approved", "canbo")
    assert [r.document_id for r in registry.servable()] == ["a"]
    assert registry.get("a").reviewed_by == "canbo"
    registry.mark_indexed({"a": 12}, "finetuned-e5")
    record = registry.get("a")
    assert record.status == "indexed" and record.chunk_count == 12 and record.embedding_model_version == "finetuned-e5"
    registry.set_status("a", "withdrawn", "admin", note="nguồn sai")
    assert registry.servable() == []


def test_supersede_keeps_history_and_audit_log(registry, tmp_path):
    registry.register(make_record(tmp_path, "old", "cũ", status="approved"), "admin")
    registry.register(make_record(tmp_path, "new", "mới", status="approved", effective_from="2026-01-01"), "admin")
    registry.mark_indexed({"old": 1, "new": 1}, "m")
    registry.supersede("old", "new", "admin")
    old, new = registry.get("old"), registry.get("new")
    assert (old.status, old.legal_status, old.superseded_by, old.effective_to) == (
        "superseded", "superseded", "new", "2026-01-01")
    assert new.supersedes == ["old"]
    actions = [entry["action"] for entry in registry.audit_log()]
    assert actions[0] == "supersede" and actions.count("register") == 2   # BR-08: mọi thay đổi có log


def test_metadata_validation_and_corpus_version(registry, tmp_path):
    with pytest.raises(ValueError):
        make_record(tmp_path, "x", access_scope="secret")
    with pytest.raises(ValueError):
        make_record(tmp_path, "y", effective_from="01/01/2025")       # ngày phải theo ISO 8601
    registry.register(make_record(tmp_path, "a", status="approved"), "admin")
    before = corpus_version(registry.servable())
    original_hash = registry.get("a").file_hash
    registry.update_metadata("a", "admin", legal_status="verified_current", file_hash="hack", status="indexed")
    record = registry.get("a")
    assert record.legal_status == "verified_current"
    assert record.file_hash == original_hash and record.status == "approved"   # trường được bảo vệ không đổi
    assert corpus_version(registry.servable()) != before             # đổi metadata lọc → phải đánh chỉ mục lại


LAW_TEXT = "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về quyền và nghĩa vụ của các bên trong quan hệ dân sự.\n"


def test_missing_source_reuses_indexed_chunks_of_same_version(registry, tmp_path):
    from src.knowledge.corpus import build_chunks

    record = registry.register(make_record(tmp_path, "luat", LAW_TEXT, status="approved"), "admin")
    indexed = build_chunks(registry, registry.servable(), log=lambda _: None)
    registry.update_metadata("luat", "admin", legal_status="amended")
    registry.resolve_path(record).unlink()                       # tệp gốc bị mất, chỉ mục còn

    messages = []
    chunks = build_chunks(registry, registry.servable(), indexed=lambda: indexed, log=messages.append)
    assert [c.page_content for c in chunks] == [c.page_content for c in indexed]
    assert all(c.metadata["legal_status"] == "amended" for c in chunks)  # metadata lấy theo registry hiện tại
    assert messages and "luat.txt" in messages[0]


def test_missing_source_without_index_copy_is_reported(registry, tmp_path):
    from src.knowledge.corpus import MissingSourceError, build_chunks

    record = registry.register(make_record(tmp_path, "luat", LAW_TEXT, status="approved"), "admin")
    registry.resolve_path(record).unlink()
    with pytest.raises(MissingSourceError, match="luat.txt"):
        build_chunks(registry, registry.servable(), indexed=lambda: [])
