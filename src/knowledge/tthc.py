"""
Thủ tục hành chính (TTHC) cấp xã: parser văn bản công bố → bản ghi có cấu trúc → chunk theo nhóm trường.

Mỗi TTHC giữ các trường riêng (trình tự, thành phần hồ sơ, thời hạn, lệ phí, căn cứ pháp lý...) thay vì một chuỗi
dài (mục 2.3.3), để bộ truy xuất trả đúng phần được hỏi và trích dẫn ghi được "mục" của thủ tục.

Parser viết cho định dạng "nội dung cụ thể của TTHC" mà UBND cấp tỉnh công bố (tiêu đề "1. Thủ tục ...",
rồi các mục "Trình tự thực hiện", "Thành phần hồ sơ"...). Văn bản trích từ PDF hay bị tách chữ ("Th ủ tục"),
nên tiêu đề mục được nhận diện sau khi bỏ khoảng trắng; nội dung giữ nguyên văn bản trích xuất.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document

try:
    from .schema import DocumentRecord, stable_chunk_id
except ImportError:
    from knowledge.schema import DocumentRecord, stable_chunk_id

# (khóa trường, nhãn hiển thị, các tiền tố đã bỏ khoảng trắng)
FIELDS = [
    ("trinh_tu", "Trình tự thực hiện", ("Trìnhtựthựchiện",)),
    ("cach_thuc", "Cách thức thực hiện", ("Cáchthứcthựchiện",)),
    ("thanh_phan_ho_so", "Thành phần hồ sơ", ("Thànhphần,sốlượnghồsơ", "Thànhphầnhồsơ")),
    ("so_luong_ho_so", "Số lượng hồ sơ", ("Sốlượnghồsơ",)),
    ("thoi_han", "Thời hạn giải quyết", ("Thờihạngiảiquyết",)),
    ("doi_tuong", "Đối tượng thực hiện", ("Đốitượngthựchiện",)),
    ("co_quan_thuc_hien", "Cơ quan thực hiện", ("Cơquanthựchiện",)),
    ("co_quan_phoi_hop", "Cơ quan phối hợp", ("Cơquanphốihợp",)),
    ("ket_qua", "Kết quả thực hiện", ("Kếtquảthựchiện",)),
    ("le_phi", "Phí, lệ phí", ("Phí,lệphí", "Lệphí", "Phí:")),
    ("mau_don", "Mẫu đơn, mẫu tờ khai", ("Tênmẫuđơn", "Mẫuđơn")),
    ("yeu_cau_dieu_kien", "Yêu cầu, điều kiện thực hiện", ("Yêucầu,điềukiện",)),
    ("can_cu_phap_ly", "Căn cứ pháp lý", ("Căncứpháplý",)),
]
FIELD_LABELS = {key: label for key, label, _ in FIELDS}

# Nhóm trường thành chunk: câu hỏi "cần giấy tờ gì" khớp chunk hồ sơ, "mất bao lâu / bao nhiêu tiền" khớp chunk
# thời hạn – lệ phí... Mỗi chunk lặp lại tên thủ tục để tự đủ nghĩa khi được truy xuất riêng.
CHUNK_GROUPS = [
    ("tong_quan", "Thông tin chung", ["doi_tuong", "co_quan_thuc_hien", "co_quan_phoi_hop", "ket_qua"]),
    ("trinh_tu", "Trình tự, cách thức thực hiện", ["trinh_tu", "cach_thuc"]),
    ("ho_so", "Thành phần, số lượng hồ sơ", ["thanh_phan_ho_so", "so_luong_ho_so"]),
    ("thoi_han_le_phi", "Thời hạn giải quyết, phí, lệ phí, mẫu đơn", ["thoi_han", "le_phi", "mau_don"]),
    ("dieu_kien", "Yêu cầu, điều kiện", ["yeu_cau_dieu_kien"]),
    ("can_cu", "Căn cứ pháp lý", ["can_cu_phap_ly"]),
]
MAX_CHUNK_CHARS = 1400

_PAGE_MARK = re.compile(r"^=== PAGE \d+ ===$|^\d{1,3}$")
_PROCEDURE_START = re.compile(r"^\d+\.(Thủtục|Đăngký|Liênthông|Giảiquyết)")
_SECTION = re.compile(r"^(I|II|III|IV|V|VI)\.Lĩnhvực(.+?)(\(|$)")
_FORM_START = "CỘNGHÒAXÃHỘICHỦNGHĨA"
_SECTION_NAMES = {"hộtịch": "Hộ tịch", "chứngthực": "Chứng thực", "nuôiconnuôi": "Nuôi con nuôi",
                  "bồithườngnhànước": "Bồi thường nhà nước"}


_ONSET = r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)"
_VOWEL_START = "aăâeêioôơuưyáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
# Lỗi trích xuất PDF tách một âm tiết thành phụ âm đầu + phần vần ("ch ứng th ực"). Phụ âm đứng riêng không phải
# một từ tiếng Việt nên ghép lại là an toàn; trừ ký hiệu điểm ("điểm b", "điểm đ") của văn bản pháp luật.
_BROKEN_SYLLABLE = re.compile(
    rf"(?<![\wđĐ])(?<!điểm )(?<!Điểm )({_ONSET}) (?=[{_VOWEL_START}])", re.IGNORECASE)


_TOKEN = re.compile(r"^([^\W\d_]+)(.*)$")
_LEADING = re.compile(r"^([^\w\s]*)([^\W\d_]+)$")   # "(phư" → "(", "phư"
_ONSET_ONLY = re.compile(rf"^{_ONSET}$", re.IGNORECASE)


def _despace(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def syllable_vocabulary(paths: list[Path]) -> set[str]:
    """Tập âm tiết lấy từ văn bản sạch (docx luật trong data/raw) — dùng để ghép âm tiết bị tách khi trích PDF."""
    try:
        from ..document_upload import extract_text
    except ImportError:
        from document_upload import extract_text
    vocab: set[str] = set()
    for path in paths:
        text = unicodedata.normalize("NFC", extract_text(path.name, path.read_bytes())).lower()
        vocab.update(re.findall(r"[^\W\d_]+", text))
    return vocab


def _merge_with_vocab(line: str, vocab: set[str]) -> str:
    """Ghép "A B" → "AB" khi "AB" là âm tiết có thật và A hoặc B không phải âm tiết ("quy ết" → "quyết").
    "cho ăn" giữ nguyên vì cả hai đều là âm tiết."""
    words = line.split(" ")
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        lead = _LEADING.match(word)
        if i + 1 < len(words) and lead and (match := _TOKEN.match(words[i + 1])):
            head, tail, rest = lead.group(2), match.group(1), match.group(2)
            merged = (head + tail).lower()
            # Phụ âm đầu đứng riêng ("c ấp") có thể trùng ký hiệu điểm "c" trong từ điển → vẫn ghép, trừ "điểm c"
            lone_onset = _ONSET_ONLY.match(head) and not (out and out[-1].lower().endswith("điểm"))
            if merged in vocab and (head.lower() not in vocab or tail.lower() not in vocab or lone_onset):
                words[i + 1] = word + tail + rest  # có thể ghép tiếp với từ sau ("Đi ều" → "Điều")
                i += 1
                continue
        out.append(word)
        i += 1
    return " ".join(out)


def repair_extraction(text: str, vocab: set[str] | None = None) -> str:
    """Chuẩn hóa Unicode (NFC) và ghép âm tiết bị tách; không đổi chữ, số hay dấu câu của nội dung.
    Có từ điển âm tiết thì ghép theo từ điển, không có thì chỉ ghép phụ âm đầu đứng riêng ("ch ứng")."""
    text = unicodedata.normalize("NFC", text)
    if vocab:
        return "\n".join(_merge_with_vocab(line, vocab) for line in text.split("\n"))
    return _BROKEN_SYLLABLE.sub(r"\1", text)


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _match_field(line: str) -> tuple[str, str] | None:
    """Dòng mở đầu một mục → (khóa trường, phần nội dung còn lại trên dòng)."""
    compact = _despace(line)
    for key, _, prefixes in FIELDS:
        if any(compact.startswith(prefix) for prefix in prefixes):
            head = line[:90]
            if ":" in head:
                return key, line[line.index(":") + 1:].strip()
            # Tiêu đề không có dấu ":" (vd. "Trình tự thực hiện") chỉ hợp lệ khi đứng riêng một dòng ngắn
            return (key, "") if len(line.strip()) <= 60 else None
    return None


def parse_procedures(text: str, level: str = "Cấp xã", vocab: set[str] | None = None) -> list[dict]:
    """Tách văn bản công bố TTHC thành danh sách thủ tục; bỏ phần mẫu đơn (tờ khai) đi kèm."""
    text = repair_extraction(text, vocab)
    lines = [line.rstrip() for line in text.split("\n") if not _PAGE_MARK.match(line.strip())]
    procedures: list[dict] = []
    section = None
    current: dict | None = None
    field_key = None
    in_form = False
    title_open = False

    def finish():
        if current is not None:
            current["fields"] = {k: _clean(v) for k, v in current["fields"].items() if _clean(v)}
            if current["fields"]:
                procedures.append(current)

    for line in lines:
        compact = _despace(line)
        if not compact:
            continue
        if section_match := _SECTION.match(compact):
            name = section_match.group(2)
            section = _SECTION_NAMES.get(name.lower(), name)
            continue
        if _PROCEDURE_START.match(compact):
            finish()
            title = re.sub(r"^\s*\d+\.\s*", "", line).strip()
            current = {"title": title, "field_area": section, "level": level, "fields": {}}
            field_key, in_form, title_open = None, False, True
            continue
        if current is None:
            continue
        if compact.startswith(_FORM_START):
            in_form = True  # mẫu tờ khai in kèm: không phải nội dung thủ tục
            continue
        matched = _match_field(line)
        if matched:
            field_key, rest = matched
            in_form = False
            title_open = False
            current["fields"].setdefault(field_key, "")
            if rest:
                current["fields"][field_key] += rest + "\n"
            continue
        if in_form:
            continue
        if title_open and field_key is None:
            current["title"] += " " + line.strip()  # tiêu đề dài xuống dòng
            continue
        if field_key:
            current["fields"][field_key] += line + "\n"
    finish()

    for proc in procedures:
        proc["title"] = re.sub(r"\s+", " ", proc["title"]).strip().rstrip(".")
        if not _despace(proc["title"]).lower().startswith(("thủtục", "liênthông")):
            proc["title"] = "Thủ tục " + proc["title"][0].lower() + proc["title"][1:]
    return _dedupe(procedures)


def _dedupe(procedures: list[dict]) -> list[dict]:
    """Văn bản công bố đôi khi in một thủ tục hai lần: giữ bản đầy đủ hơn."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for proc in procedures:
        key = _despace(proc["title"]).lower()
        size = sum(len(v) for v in proc["fields"].values())
        if key not in best:
            order.append(key)
            best[key] = proc
        elif size > sum(len(v) for v in best[key]["fields"].values()):
            best[key] = proc
    return [best[key] for key in order]


def load_procedures(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["procedures"] if isinstance(data, dict) else data


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + 1 + len(line) > max_chars:
            pieces.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        pieces.append(current)
    return pieces


def procedure_chunks(record: DocumentRecord, procedures: list[dict], start_index: int = 0) -> list[Document]:
    """Bản ghi TTHC → chunk: mỗi nhóm trường một chunk (nhóm dài thì tách theo dòng, lặp lại tiêu đề)."""
    chunks: list[Document] = []
    index = start_index
    base = record.chunk_metadata()
    for number, proc in enumerate(procedures, 1):
        procedure_id = proc.get("id") or f"{record.document_id}-{number:02d}"
        for group_key, group_label, keys in CHUNK_GROUPS:
            parts = [f"{FIELD_LABELS[k]}: {proc['fields'][k]}" for k in keys if proc["fields"].get(k)]
            if not parts:
                continue
            header = f"[TTHC {proc.get('level', 'cấp xã').lower()}: {proc['title']}] {group_label}"
            body = "\n".join(parts)
            pieces = _split_text(body, MAX_CHUNK_CHARS)
            for part_no, piece in enumerate(pieces):
                suffix = f" (phần {part_no + 1}/{len(pieces)})" if len(pieces) > 1 else ""
                metadata = {
                    **base,
                    "chunk_id": stable_chunk_id(record.version_id, index),
                    "chunk_index": index,
                    "procedure_id": procedure_id,
                    "procedure_title": proc["title"],
                    "procedure_code": proc.get("code"),
                    "procedure_field": group_key,
                    "field_area": proc.get("field_area") or record.field_area,
                    "article": f"{proc['title']} — {group_label}{suffix}",
                    "article_reference": f"Mục: {group_label}{suffix}",
                }
                chunks.append(Document(page_content=f"{header}{suffix}\n{piece}", metadata=metadata))
                index += 1
    return chunks
