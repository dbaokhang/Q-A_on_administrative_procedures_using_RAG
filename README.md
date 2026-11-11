# Trợ lý ảo hỗ trợ chính quyền địa phương cấp cơ sở

**Đồ án chuyên ngành — Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn.**
Sinh viên: Dương Bảo Khang, Mai Vũ Trung Tín · Giảng viên hướng dẫn: TS. Phan Tấn Quốc.

Trợ lý ảo trả lời câu hỏi bằng tiếng Việt về **thủ tục hành chính (TTHC) cấp xã** và **văn bản quy phạm pháp luật
(VBQPPL)**. Hệ thống dùng kiến trúc Hybrid RAG: BM25 kết hợp vector ngữ nghĩa, hợp nhất bằng RRF, rồi tái xếp hạng bằng Cross-Encoder.
Mỗi câu trả lời kèm trích dẫn đã được kiểm tra (tên văn bản, số hiệu, Điều hoặc mục, trạng thái hiệu lực). Khi kho
tài liệu chưa đủ căn cứ, hệ thống nói rõ là chưa đủ căn cứ thay vì tự suy đoán. Thiết kế bám theo Chương 2 của báo cáo giữa kỳ; phần đối chiếu
từng yêu cầu với mã nguồn nằm trong [ARCHITECTURE.md](ARCHITECTURE.md).

> Đây là nguyên mẫu nghiên cứu. Câu trả lời chỉ hỗ trợ tra cứu, không thay thế hướng dẫn hay quyết định của cán bộ
> có thẩm quyền. Dữ liệu TTHC hiện có là **dữ liệu mẫu chưa xác minh hiệu lực** (xem mục Dữ liệu).

## Chức năng chính

- **Hỏi đáp có căn cứ.** Truy xuất lai, lọc theo quyền và hiệu lực, Cross-Encoder, sau đó LLM trả lời với mã nguồn `[S#]`.
  Lớp kiểm soát chặn những câu trả lời trích mã nguồn không tồn tại hoặc nêu số Điều, số hiệu không có trong ngữ cảnh; câu trả lời được sinh
  lại một lần, nếu vẫn sai thì trả lời an toàn kèm mã tra cứu.
- **Hiểu câu hỏi về thủ tục.** Từ điển chuyển cách nói thông thường sang thuật ngữ ("làm lại giấy khai sinh" → "đăng ký lại
  khai sinh"). Bộ phân loại ý định chọn đúng mục của TTHC: thành phần hồ sơ, thời hạn – lệ phí, điều kiện, trình tự, căn cứ pháp lý.
  Câu hỏi quá chung ("Làm hồ sơ thế nào?") được hỏi lại kèm gợi ý thủ tục.
- **Bảy trạng thái phản hồi** theo Bảng 2.14: có nguồn, cần làm rõ, chưa đủ căn cứ, không đủ quyền, lỗi tạm thời, yêu cầu
  không hợp lệ, lời chào.
- **Phân quyền** người dân / cán bộ / quản trị. Tài liệu nội bộ bị lọc ngay trong truy vấn BM25 và Qdrant, nên không bao giờ đi vào
  ngữ cảnh của phiên công khai. Dữ liệu cá nhân (số định danh, số điện thoại, email) được che trước khi gửi tới LLM và trước khi ghi log.
- **Quản trị kho tri thức.** Nạp tài liệu → chờ kiểm tra → duyệt → đánh chỉ mục → rút hoặc thay thế, kèm nhật ký thao tác.
  Quản trị viên sửa được metadata hiệu lực. Chỉ mục Qdrant dựng theo phiên bản và chỉ embed phần mới.
- **Giám sát và đánh giá.** Nhật ký truy vấn có `correlation_id` và độ trễ từng bước; người dùng đánh giá 👍/👎 kèm lý do. Benchmark
  định dạng Vi-CommuneQA đo Recall@5, Precision@3, MRR, NDCG@5, ROUGE-1/2/L, BLEU-4, BERTScore, độ chính xác trạng thái,
  tỷ lệ trích đúng nguồn, P50/P95.

## Cài đặt và chạy

Yêu cầu: Python 3.11. Cần một trong hai LLM: Gemini (`GOOGLE_API_KEY`) hoặc [Ollama](https://ollama.com) chạy trên máy
(`ollama pull qwen3:8b`).

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                     # điền GOOGLE_API_KEY hoặc LLM_PROVIDER=ollama

# Tài khoản cán bộ / quản trị (người dân không cần đăng nhập). Đặt APP_SECRET trong .env khi chạy API.
python -m src.auth add-user canbo01 officer
python -m src.auth add-user quantri01 admin

streamlit run src/app.py                                 # → http://localhost:8501
uvicorn src.api:app                                      # → http://localhost:8000/docs
```

Lần chạy đầu, hệ thống tạo registry từ `data/` (`src/knowledge/catalog.py`), tải mô hình embedding từ Hugging Face rồi
dựng chỉ mục Qdrant trong `qdrant_data/`. Khoảng 3.800 vector văn bản luật được lấy lại từ `faiss_index_local/`, chỉ phải embed
phần TTHC mới. Qdrant cục bộ chỉ cho **một** tiến trình mở thư mục dữ liệu. Muốn chạy Streamlit, API và đánh giá cùng lúc
thì dùng Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`, `QDRANT_URL=http://localhost:6333`).

### API

```http
POST /chat
Authorization: Bearer <token>          # không có = người dân
{"question": "Đăng ký khai sinh cần những giấy tờ gì?", "channel": "web", "history": []}
```

Phản hồi có `answer`, `status`, `sources` (chỉ các nguồn được trích, mỗi nguồn gồm tên, số hiệu, vị trí, trạng thái hiệu lực
và đoạn trích), `related_sources`, `clarification_options`, `notes`, `warnings`, `correlation_id` và `timings_ms`.
Các endpoint khác: `POST /auth/login`, `POST /feedback`, `GET /health`, `GET /admin/documents`,
`POST /admin/documents/{id}/status`, `POST /admin/reindex`, `GET /admin/metrics`, `GET /admin/audit`.

## Dữ liệu

| Thành phần | Quy mô | Nguồn |
|:--|:--|:--|
| VBQPPL | 10 văn bản: BLDS 2015, BLLĐ 2019, Luật Đất đai 2024, Luật Doanh nghiệp 2020, Luật HN&GĐ 2014, Luật Đường bộ 2024, Luật sửa đổi BLHS 2025, NĐ 100/2019, 168/2024, 238/2026 | `data/raw/` |
| TTHC cấp xã | 32 thủ tục (hộ tịch 18, chứng thực 11, nuôi con nuôi 3) | Văn bản công bố TTHC cấp xã trên Cổng TTĐT TP. Hải Phòng, trích bằng `src/knowledge/build_tthc.py` |
| Tài liệu nội bộ | 1 tài liệu giả lập để kiểm thử phân quyền | `data/internal/` |
| Benchmark | 23 câu mẫu định dạng Vi-CommuneQA | `data/benchmark/vi_communeqa_sample.jsonl` |

Dữ liệu TTHC được đánh dấu "chưa xác minh hiệu lực": căn cứ pháp lý trong văn bản nguồn chưa nêu Nghị định
120/2025/NĐ-CP (chính quyền địa phương hai cấp, từ 01/7/2025), và mức lệ phí do HĐND cấp tỉnh quy định nên khác nhau
giữa các địa phương. Quy mô dữ liệu trong báo cáo Chương 1 (1.250 TTHC, 350 VBQPPL, 520 câu) **chưa** có trong repo.

## Đánh giá

```bash
python evaluation/run_benchmark.py --retrieval-only     # chỉ truy xuất, không gọi LLM
python evaluation/run_benchmark.py                      # đầy đủ (gọi LLM); --bertscore cần: pip install bert-score
python evaluation/run_golden_set.py                     # 25 câu văn bản luật có đáp án kiểm tra thủ công
python evaluation/evaluate_finetuned.py                 # RAGAS trên 57 câu luật chưa thấy khi huấn luyện
pytest -q                                               # 135 kiểm thử đơn vị, không cần tải mô hình
```

**Kết quả trên benchmark mẫu** (05/11/2026, 23 câu; [`evaluation/results_benchmark_2026-11-05.json`](evaluation/results_benchmark_2026-11-05.json)).
Cấu hình: fine-tuned E5 + BM25, hợp nhất RRF, Cross-Encoder, Qdrant cục bộ, LLM `llama3.1:8b` qua Ollama, CPU.

| Nhóm | Chỉ số | Kết quả | Mục tiêu báo cáo |
|:--|:--|:--:|:--:|
| Truy xuất (17 câu có đoạn chuẩn) | Recall@5 | 0,941 | ≥ 0,90 |
| | Precision@3 · MRR@10 · NDCG@5 | 0,392 · 0,971 · 0,933 | — |
| Kiểm soát (23 câu) | Đúng trạng thái (trả lời / hỏi lại / từ chối / không đủ quyền) | 23/23 | — |
| | Trích đúng đoạn chuẩn (17 câu cần trả lời) | 17/17 | ≥ 95% |
| | Chứa đủ ý chính | 15/17 | — |
| | Lộ tài liệu nội bộ cho phiên công khai | 0 | 0 |
| | Phân loại ý định đúng | 19/20 | — |
| Sinh (17 câu) | ROUGE-1 · ROUGE-2 · ROUGE-L · BLEU-4 | 0,592 · 0,533 · 0,576 · 0,403 | — |
| Độ trễ | Truy xuất (BM25 + dense + rerank), trung bình | ≈ 2,7 s (rerank 2,2 s) | P95 < 1 s |
| | Đầu-cuối P50 · P95 | 7,5 s · 24,8 s | < 5 s |

Đọc kết quả cho đúng:

- **Benchmark nhỏ và do nhóm tự soạn.** Đoạn chuẩn và câu trả lời tham chiếu lấy nguyên văn từ dữ liệu TTHC; các câu về luật dùng đáp án đã kiểm tra thủ công của
  `golden_set.json`. Đây là phép kiểm tra tính đúng của pipeline, **không** thay cho Vi-CommuneQA 520 câu như báo cáo dự kiến.
- **Precision@3 thấp là do cách tính.** Mỗi câu chỉ có 1–2 đoạn chuẩn nên điểm tối đa thường là 1/3 hoặc 2/3.
- **Hai câu thiếu ý chính.** Câu "khai tử nộp ở đâu" trả lời đúng nhưng còn chung chung, vì mục cơ quan thực hiện không vào được top 5. Câu hỏi
  tiếng Anh thì `llama3.1:8b` trả lời bằng tiếng Việt.
- **Độ trễ chưa đạt mục tiêu.** Nguyên nhân là Cross-Encoder và LLM chạy trên CPU của một máy còn rất ít RAM trống. Có thể giảm `RERANK_POOL`, chạy GPU
  hoặc dùng Gemini.
- **BERTScore và RAGAS chưa được chạy** trong lần đo này. Chạy bằng `--bertscore` và `evaluation/evaluate_finetuned.py`.

## Cấu trúc thư mục

```
src/
├── app.py, admin_page.py        # Giao diện Web: hỏi đáp + quản trị tài liệu
├── api.py                       # FastAPI: /chat, /feedback, /auth, /admin
├── assistant.py                 # Bộ điều phối (mã giả answer_question mục 2.2.2)
├── guardrails.py                # Kiểm soát đầu vào, làm rõ, trích dẫn, prompt injection
├── intent.py                    # Phân loại ý định theo nhãn Vi-CommuneQA
├── retrieval_hybrid.py          # BM25 + dense + RRF + Cross-Encoder
├── vector_store.py              # Qdrant cục bộ / FAISS, đánh chỉ mục theo phiên bản
├── auth.py, telemetry.py        # Vai trò & token; nhật ký truy vấn & phản hồi
├── chunker.py, legal_glossary.py, embedder.py, llm_chain.py, observability.py, canned_replies.py
└── knowledge/                   # schema, registry, catalog (seed), corpus, tthc parser, filters
data/
├── raw/          # 10 VBQPPL (.docx)
├── tthc/         # TTHC cấp xã (JSON) + văn bản nguồn (sources/)
├── internal/     # tài liệu nội bộ giả lập
├── registry/     # documents.json (registry tài liệu)
├── benchmark/    # Vi-CommuneQA mẫu
└── qa/, triplets/  # dữ liệu huấn luyện / đánh giá embedding (dự án gốc)
evaluation/       # run_benchmark.py, generation_metrics.py, RAGAS, golden set
training/         # huấn luyện E5 fine-tune và Custom Transformer
tests/            # kiểm thử đơn vị
```

## Nguồn gốc

Dự án phát triển tiếp từ [Leanhoangy/vietnamese-rag-chatbot](https://github.com/Leanhoangy/vietnamese-rag-chatbot)
(giấy phép MIT, xem [LICENSE](LICENSE)). Các phần kế thừa gồm chunker văn bản luật, mô hình embedding fine-tune
([Hoangy2005/vietnamese-legal-embedder](https://huggingface.co/Hoangy2005/vietnamese-legal-embedder)), Custom
Transformer, dữ liệu huấn luyện và bộ đánh giá RAGAS / golden set.
