import json
import os
from pathlib import Path

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # USE_FINETUNED có thể nằm trong .env; không phụ thuộc thứ tự import

CUSTOM_MODEL_PATH = Path("models/custom-transformer/model.pt")
CUSTOM_CONFIG_PATH = Path("models/custom-transformer/config.json")
CUSTOM_TOKENIZER_PATH = Path("models/custom-tokenizer/vocab.json")

FINETUNED_MODEL_PATH = Path("models/finetuned-embedder")
FINETUNED_CONFIG_PATH = FINETUNED_MODEL_PATH / "config.json"


class CustomTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper cho custom transformer."""

    def __init__(self, model, tokenizer, max_len: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _encode_text(self, text: str) -> list[float]:
        enc = self.tokenizer.encode_plus(text, self.max_len)
        input_ids = torch.tensor([enc["input_ids"]])
        attention_mask = torch.tensor([enc["attention_mask"]])
        vec = self.model.encode(input_ids, attention_mask)
        return vec[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode_text(text)


def _load_custom_transformer() -> CustomTransformerEmbeddings | None:
    if not (CUSTOM_MODEL_PATH.exists() and CUSTOM_CONFIG_PATH.exists() and CUSTOM_TOKENIZER_PATH.exists()):
        return None
    from src.custom_transformer.tokenizer import LegalTokenizer
    from src.custom_transformer.architecture import LegalTransformerEncoder

    with open(CUSTOM_CONFIG_PATH) as f:
        cfg = json.load(f)

    tokenizer = LegalTokenizer()
    tokenizer.load(str(CUSTOM_TOKENIZER_PATH))

    model = LegalTransformerEncoder(
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_len=cfg["max_len"],
    )
    state = torch.load(str(CUSTOM_MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return CustomTransformerEmbeddings(model, tokenizer, cfg["max_len"])


def _download_finetuned_model() -> None:
    """Download fine-tuned model từ HuggingFace Hub nếu chưa có."""
    if _has_valid_finetuned_model():
        return
    try:
        from huggingface_hub import snapshot_download
        print("Downloading fine-tuned model from HuggingFace Hub...")
        snapshot_download(
            repo_id="Hoangy2005/vietnamese-legal-embedder",
            local_dir=str(FINETUNED_MODEL_PATH),
        )
        print("Download complete.")
    except Exception as e:
        print(f"Could not download fine-tuned model: {e}")


def _has_valid_finetuned_model() -> bool:
    if not FINETUNED_MODEL_PATH.exists() or not FINETUNED_CONFIG_PATH.exists():
        return False
    try:
        with open(FINETUNED_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return "model_type" in config


def _hf(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# EMBEDDING_MODEL chọn mô hình để so sánh trên benchmark (Bảng 2.8): auto | finetuned-e5 | e5-base | bge-m3 | custom.
# auto (mặc định) giữ thứ tự ưu tiên cũ: custom transformer → fine-tuned e5 → base e5.
# EMBEDDING_MODEL_KEY đi vào tên collection: vector của các mô hình khác nhau không bao giờ bị trộn.
# Set USE_FINETUNED=true để bỏ qua custom transformer (dùng khi evaluate fine-tuned e5)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "auto").strip().lower()
use_finetuned = os.environ.get("USE_FINETUNED", "false").lower() == "true"

if EMBEDDING_MODEL == "bge-m3":
    # ~2,3 GB, 1024 chiều — chỉ bật khi máy đủ RAM
    print("Loading BGE-M3 embedding model...")
    embeddings, EMBEDDING_MODEL_KEY = _hf("BAAI/bge-m3"), "bge-m3"
elif EMBEDDING_MODEL == "e5-base":
    print("Using base embedding model (multilingual-e5-base).")
    embeddings, EMBEDDING_MODEL_KEY = _hf("intfloat/multilingual-e5-base"), "e5-base"
else:
    _download_finetuned_model()
    custom = None if (use_finetuned or EMBEDDING_MODEL == "finetuned-e5") else _load_custom_transformer()
    if EMBEDDING_MODEL == "custom" and custom is None:
        raise RuntimeError("EMBEDDING_MODEL=custom nhưng chưa có models/custom-transformer/ — hãy train trước.")
    if custom is not None:
        print("Loading custom transformer embedding model...")
        embeddings, EMBEDDING_MODEL_KEY = custom, "custom-transformer"
    elif _has_valid_finetuned_model():
        print("Loading fine-tuned embedding model...")
        embeddings, EMBEDDING_MODEL_KEY = _hf(str(FINETUNED_MODEL_PATH)), "finetuned-e5"
    else:
        print("Using base embedding model (multilingual-e5-base).")
        embeddings, EMBEDDING_MODEL_KEY = _hf("intfloat/multilingual-e5-base"), "e5-base"
