"""
Dựng tệp dữ liệu TTHC (JSON) từ văn bản công bố nội dung TTHC của UBND cấp tỉnh (PDF/DOCX/TXT).

    python -m src.knowledge.build_tthc data/tthc/sources/haiphong_tthc_cap_xa.pdf \
        --out data/tthc/haiphong_tthc_cap_xa.json --source-url <URL> --publisher "UBND TP. Hải Phòng"

Chỉ lấy phần thủ tục thuộc thẩm quyền UBND cấp xã: từ tiêu đề có "THẨM QUYỀN GIẢI QUYẾT CỦA UBND CẤP XÃ"
tới tiêu đề phần kế tiếp ("E. ..."). Kết quả là dữ liệu thô cần quản trị viên rà soát trước khi duyệt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_upload import extract_text  # noqa: E402
from src.knowledge.schema import now_iso  # noqa: E402
from src.knowledge.tthc import _despace, parse_procedures, syllable_vocabulary  # noqa: E402

_COMMUNE_PART = re.compile(r"THẨMQUYỀNGIẢIQUYẾTCỦAUBNDCẤPXÃ")
_NEXT_PART = re.compile(r"^[E-H]\.THỦTỤC")


def commune_section(text: str) -> str:
    lines = text.split("\n")
    # Tiêu đề phần thường xuống dòng ("... THẨM QUYỀN GIẢI QUYẾT" / "CỦA UBND CẤP XÃ") → ghép hai dòng liền nhau
    start = next((i for i in range(len(lines))
                  if _COMMUNE_PART.search(_despace(" ".join(lines[i:i + 2])))), None)
    if start is None:
        return text
    end = next((i for i in range(start + 1, len(lines)) if _NEXT_PART.match(_despace(lines[i]))), len(lines))
    return "\n".join(lines[start:end])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--publisher")
    args = parser.parse_args()

    path = Path(args.source)
    text = extract_text(path.name, path.read_bytes())
    vocab = syllable_vocabulary(sorted((PROJECT_ROOT / "data" / "raw").glob("*.docx")))
    procedures = parse_procedures(commune_section(text), vocab=vocab)
    for number, proc in enumerate(procedures, 1):
        proc["id"] = f"{Path(args.out).stem}-{number:02d}"
        proc["code"] = None  # văn bản nguồn không ghi mã TTHC; bổ sung khi đối chiếu CSDL quốc gia về TTHC
    payload = {
        "source": {
            "file": path.name,
            "url": args.source_url,
            "publisher": args.publisher,
            "extracted_at": now_iso(),
            "note": "Trích tự động từ văn bản công bố; nội dung giữ nguyên văn bản trích xuất, cần rà soát.",
        },
        "procedures": procedures,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(procedures)} thủ tục → {out}")
    for proc in procedures:
        print(f"  [{proc['field_area']}] {proc['title']}  ({', '.join(proc['fields'])})")


if __name__ == "__main__":
    main()
