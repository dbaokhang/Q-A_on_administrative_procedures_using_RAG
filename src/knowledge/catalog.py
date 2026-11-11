"""
Danh mục khởi tạo kho tri thức (seed): 10 VBQPPL có sẵn, bộ TTHC cấp xã mẫu và một tài liệu nội bộ giả lập.

Số hiệu, ngày ban hành và ngày hiệu lực lấy từ chính nội dung từng văn bản (phần đầu và điều khoản
"Hiệu lực thi hành"). legal_status mặc định "unverified": việc xác minh văn bản sửa đổi, thay thế sau này
là trách nhiệm của quản trị viên tri thức (BR-01, BR-03), không suy đoán trong mã nguồn.

    python -m src.knowledge.catalog          # tạo data/registry/documents.json nếu chưa có
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import DOC_LABELS, DOC_NOTES  # noqa: E402
from src.knowledge.registry import DocumentRegistry, RegistryError  # noqa: E402
from src.knowledge.schema import DocumentRecord  # noqa: E402

QUOC_HOI, CHINH_PHU = "Quốc hội", "Chính phủ"

# (tệp, mã tài liệu, số hiệu, cơ quan ban hành, ngày ban hành, hiệu lực từ, lĩnh vực, legal_status)
LAWS = [
    ("Luatdansu.docx", "bo-luat-dan-su-2015", "91/2015/QH13", QUOC_HOI, "2015-11-24", "2017-01-01", "Dân sự", "unverified"),
    ("Luatdatdai.docx", "luat-dat-dai-2024", "31/2024/QH15", QUOC_HOI, "2024-01-18", "2025-01-01", "Đất đai", "unverified"),
    ("Luatdoanhnghiep.docx", "luat-doanh-nghiep-2020", "59/2020/QH14", QUOC_HOI, "2020-06-17", "2021-01-01", "Doanh nghiệp", "unverified"),
    ("Luatduongbo2024.docx", "luat-duong-bo-2024", "35/2024/QH15", QUOC_HOI, "2024-06-27", "2025-01-01", "Giao thông", "unverified"),
    ("Luathinhsu.docx", "luat-sua-doi-bo-luat-hinh-su-2025", "86/2025/QH15", QUOC_HOI, "2025-06-25", "2025-07-01", "Hình sự", "unverified"),
    ("Luathonnhan.docx", "luat-hon-nhan-gia-dinh-2014", "52/2014/QH13", QUOC_HOI, "2014-06-19", "2015-01-01", "Hôn nhân và gia đình", "unverified"),
    ("Luatlaodong.docx", "bo-luat-lao-dong-2019", "45/2019/QH14", QUOC_HOI, "2019-11-20", "2021-01-01", "Lao động", "unverified"),
    # Chương II (đường bộ) bị NĐ 168/2024 thay thế từ 01/01/2025 (Điều 53, 54 NĐ 168) — xem chunker.SUPERSEDED_ARTICLES
    ("Nghidinh 100-2019.docx", "nghi-dinh-100-2019", "100/2019/NĐ-CP", CHINH_PHU, "2019-12-30", "2020-01-01", "Giao thông", "partially_superseded"),
    # Một số điều được NĐ 238/2026 sửa đổi, bổ sung từ 15/08/2026
    ("Nghidinh_168_2024.docx", "nghi-dinh-168-2024", "168/2024/NĐ-CP", CHINH_PHU, "2024-12-26", "2025-01-01", "Giao thông", "amended"),
    ("Nghidinh_238_2026.docx", "nghi-dinh-238-2026", "238/2026/NĐ-CP", CHINH_PHU, "2026-06-26", "2026-08-15", "Giao thông", "unverified"),
]

TTHC_SOURCE_URL = ("https://cdn.haiphong.gov.vn/gov-hpg/6846/tintuc/2025/9/"
                   "tthc-cap-xa638309686635908721638937146928842646.pdf")
TTHC_NOTE = (
    "Dữ liệu mẫu: trích tự động từ văn bản công bố nội dung TTHC thuộc thẩm quyền UBND cấp xã đăng trên Cổng thông "
    "tin điện tử TP. Hải Phòng. Căn cứ pháp lý trong văn bản chưa nêu Nghị định 120/2025/NĐ-CP (phân định thẩm quyền "
    "chính quyền địa phương hai cấp, từ 01/7/2025); mức lệ phí do HĐND cấp tỉnh quy định nên khác nhau giữa địa "
    "phương. Cần đối chiếu Cơ sở dữ liệu quốc gia về TTHC trước khi dùng chính thức."
)


def seed_records() -> list[DocumentRecord]:
    records = []
    for file_name, doc_id, number, body, issued, effective, area, legal_status in LAWS:
        label = DOC_LABELS[file_name]
        records.append(DocumentRecord(
            document_id=doc_id, title=label, document_type="law", file_path=f"data/raw/{file_name}",
            document_number=number, issuing_body=body, issued_date=issued, effective_from=effective,
            legal_status=legal_status, access_scope="public", field_area=area, status="approved",
            notes=DOC_NOTES.get(label), reviewed_by="seed",
            source_note="Văn bản có sẵn trong kho dữ liệu của dự án gốc (data/raw).",
        ))
    records.append(DocumentRecord(
        document_id="tthc-cap-xa-haiphong", title="Bộ TTHC thuộc thẩm quyền UBND cấp xã (TP. Hải Phòng công bố)",
        document_type="administrative_procedure", file_path="data/tthc/haiphong_tthc_cap_xa.json",
        issuing_body="UBND TP. Hải Phòng", legal_status="unverified", access_scope="public",
        field_area="Hộ tịch; Chứng thực; Nuôi con nuôi", source_url=TTHC_SOURCE_URL, status="approved",
        notes=TTHC_NOTE, reviewed_by="seed", source_note="Bản gốc: data/tthc/sources/haiphong_tthc_cap_xa.pdf",
    ))
    records.append(DocumentRecord(
        document_id="noi-bo-huong-dan-tiep-nhan-mau", title="Hướng dẫn nội bộ tiếp nhận hồ sơ hộ tịch (MẪU GIẢ LẬP)",
        document_type="internal_guide", file_path="data/internal/huong_dan_noi_bo_mau.md",
        issuing_body="Nhóm đồ án (giả lập)", legal_status="unverified", access_scope="internal",
        field_area="Nghiệp vụ một cửa", status="approved", reviewed_by="seed",
        notes="Tài liệu giả lập phục vụ kiểm thử phân quyền; không phải văn bản của cơ quan nhà nước.",
    ))
    return records


def seed(registry: DocumentRegistry | None = None, actor: str = "seed") -> DocumentRegistry:
    registry = registry or DocumentRegistry()
    existing = {r.document_id for r in registry.all()}
    for record in seed_records():
        if record.document_id in existing:
            continue
        try:
            registry.register(record, actor)
        except RegistryError as e:
            print(f"Bỏ qua {record.document_id}: {e}")
    return registry


if __name__ == "__main__":
    reg = seed()
    for rec in reg.all():
        print(f"{rec.status:10} {rec.access_scope:8} {rec.legal_status:22} {rec.document_id}")
