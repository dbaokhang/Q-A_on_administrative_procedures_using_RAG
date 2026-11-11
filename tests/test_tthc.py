from src.knowledge.schema import DocumentRecord
from src.knowledge.tthc import parse_procedures, procedure_chunks, repair_extraction

PUBLICATION = """D. THỦ TỤC HÀNH CHÍNH THUỘC THẨM QUYỀN GIẢI QUYẾT CỦA
UBND CẤP XÃ
I. Lĩnh vực hộ tịch
1. Thủ tục đăng ký khai sinh
Trình tự thực hiện
- Người có yêu cầu nộp hồ sơ tại Ủy ban nhân dân cấp xã có thẩm quyền.
Thành phần hồ sơ:
* Giấy tờ phải nộp:
- Tờ khai đăng ký khai sinh theo mẫu.
=== PAGE 2 ===
2
- Bản chính Giấy chứng sinh.
Số lượng hồ sơ: 01 bộ
Thời hạn giải quyết : Ngay trong ngày tiếp nhận yêu cầu.
Đối tư ợng th ực hi ện thủ tục hành chính: Cá nhân
Lệ phí: Miễn lệ phí đối với trường hợp khai sinh đúng hạn.
Căn cứ pháp lý:
- Luật Hộ tịch năm 2014;
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
TỜ KHAI ĐĂNG KÝ KHAI SINH
Họ, chữ đệm, tên người yêu cầu: ........
II. Lĩnh vực chứng thực (01 thủ tục)
1. Th ủ tục ch ứng th ực bản sao từ bản chính giấy tờ, văn bản do cơ quan, tổ
chức có thẩm quyền cấp
Trình tự thực hiện:
- Người yêu cầu xuất trình bản chính.
Lệ phí: 2.000 đồng/trang.
"""


def test_parses_procedures_fields_and_sections():
    procedures = parse_procedures(PUBLICATION)
    assert [p["title"] for p in procedures] == [
        "Thủ tục đăng ký khai sinh",
        "Thủ tục chứng thực bản sao từ bản chính giấy tờ, văn bản do cơ quan, tổ chức có thẩm quyền cấp",
    ]
    birth, copy = procedures
    assert birth["field_area"] == "Hộ tịch" and copy["field_area"] == "Chứng thực"
    assert "Bản chính Giấy chứng sinh" in birth["fields"]["thanh_phan_ho_so"]   # nối qua ngắt trang
    assert birth["fields"]["thoi_han"] == "Ngay trong ngày tiếp nhận yêu cầu."
    assert birth["fields"]["doi_tuong"] == "Cá nhân"                           # tiêu đề mục bị tách chữ
    assert "TỜ KHAI" not in birth["fields"]["can_cu_phap_ly"]                  # bỏ mẫu tờ khai in kèm
    assert copy["fields"]["le_phi"] == "2.000 đồng/trang."


def test_repair_extraction_merges_broken_syllables_only():
    vocab = {"quyết", "người", "cho", "ăn", "giải", "quy", "định"}
    assert repair_extraction("giải quy ết", vocab) == "giải quyết"
    assert repair_extraction("(ngư ời", vocab) == "(người"
    assert repair_extraction("cho ăn", vocab) == "cho ăn"           # hai từ thật: không ghép
    assert repair_extraction("quy định", vocab) == "quy định"
    # Không có từ điển: chỉ ghép phụ âm đầu đứng riêng, giữ ký hiệu "điểm b"
    assert repair_extraction("ch ứng th ực") == "chứng thực"
    assert repair_extraction("điểm b ản") == "điểm b ản"


def test_procedure_chunks_carry_field_metadata():
    record = DocumentRecord(document_id="tthc-test", title="Bộ TTHC mẫu", document_type="administrative_procedure",
                            file_path="x.json", file_hash="ab" * 32)
    chunks = procedure_chunks(record, parse_procedures(PUBLICATION))
    birth_groups = [c.metadata["procedure_field"] for c in chunks if c.metadata["procedure_id"] == "tthc-test-01"]
    assert birth_groups == ["tong_quan", "trinh_tu", "ho_so", "thoi_han_le_phi", "can_cu"]
    ho_so = next(c for c in chunks if c.metadata["procedure_field"] == "ho_so")
    assert ho_so.page_content.startswith("[TTHC cấp xã: Thủ tục đăng ký khai sinh] Thành phần, số lượng hồ sơ")
    assert ho_so.metadata["article_reference"] == "Mục: Thành phần, số lượng hồ sơ"
    assert ho_so.metadata["access_scope"] == "public" and ho_so.metadata["version_id"] == record.version_id
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    # Khóa ổn định: dựng lại cùng phiên bản cho cùng chunk_id (nạp lại không tạo bản trùng)
    again = procedure_chunks(record, parse_procedures(PUBLICATION))
    assert [c.metadata["chunk_id"] for c in again] == [c.metadata["chunk_id"] for c in chunks]
