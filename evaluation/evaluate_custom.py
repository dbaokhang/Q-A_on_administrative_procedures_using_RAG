"""Run evaluation only — skips training steps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.rag_evaluator import RAGEvaluator
from src.embedder import embeddings
from src.assistant import build_assistant
qa_chain = build_assistant()  # cùng giao diện invoke() với qa_chain cũ

evaluator = RAGEvaluator(
    qa_chain=qa_chain,
    embeddings=embeddings,
    test_set_path="data/qa/test_set.json",
)
evaluator.run_full_evaluation(limit=10)
