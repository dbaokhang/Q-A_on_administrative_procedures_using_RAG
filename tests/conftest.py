"""Test không cần tải mô hình (embedding, cross-encoder, LLM) hay đọc data/raw: dùng kho vector giả, LLM giả,
Qdrant in-memory và thư mục tạm cho registry / log."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
