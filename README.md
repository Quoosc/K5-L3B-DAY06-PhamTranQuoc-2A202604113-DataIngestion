# Lab06 — Data Ingestion Engineering

**Phạm Trấn Quốc · 2A202604113 · L3B**

So sánh ba chiến lược chia đoạn và lọc metadata trên tài liệu Python 3.12.10. Pipeline dùng E5 384 chiều và Qdrant local với cosine exact search. Cần mạng để cài thư viện và tải model lần đầu. Không cần API key.

| Nội dung | Đường dẫn |
|---|---|
| Học viên và báo cáo | [STUDENT.md](STUDENT.md), [báo cáo cá nhân](report/individual/2A202604113_PhamTranQuoc.md) |
| Dữ liệu, nguồn và giấy phép | [Manifest](data/manifest.json), [giấy phép CPython](data/licenses/) |
| Cấu hình và benchmark | [A/B/C](configs/strategies.json), [30 câu và gold gốc](benchmark/questions.jsonl), [giao thức](benchmark/protocol.md), [xác nhận và đính chính bằng chứng](benchmark/human_review.json) |
| Thí nghiệm và kết quả | [Nhật ký](artifacts/experiment_log.csv), [dev](results/metrics/dev.json), [test](results/metrics/test.json), [trace](results/traces/), [audit](results/audits/) |

## Cài đặt

Môi trường đã kiểm chứng là Windows, Python 3.12.10 và CPU. Từ thư mục repo, chạy:

```powershell
$env:PYTHONUTF8 = "1"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe scripts/fetch_model.py
.\.venv\Scripts\python.exe -m src.run validate
.\.venv\Scripts\python.exe -m pytest -q
```

Model `intfloat/multilingual-e5-small` được tải theo revision và checksum trong [download manifest](artifacts/model_download.json). Corpus dùng để chạy benchmark đã có sẵn trong [pages.jsonl](data/normalized/pages.jsonl). Repo giữ RST gốc. Bản văn bản đầy đủ của từng nguồn được tạo từ RST khi cần đối chiếu offset.

## Chạy lại thí nghiệm

Dựng lại bản văn bản nguồn từ RST để verifier kiểm tra offset, rồi chạy với database và output riêng:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_corpus.py --offline
.\.venv\Scripts\python.exe benchmark/build_benchmark.py
.\.venv\Scripts\python.exe -m src.run ingest --output results/reproduction --db .local/qdrant-reproduction
.\.venv\Scripts\python.exe -m src.run dev --output results/reproduction --db .local/qdrant-reproduction
.\.venv\Scripts\python.exe -m src.run test --output results/reproduction --db .local/qdrant-reproduction
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Thí nghiệm gốc chạy theo thứ tự ingest, dev, khóa cấu hình, test. Lệnh tái lập dùng [cấu hình đã khóa](artifacts/config_freeze.json) và không ghi đè trace hoặc metric gốc. [build_benchmark.py](benchmark/build_benchmark.py) chứa thiết kế câu hỏi và quy tắc chọn vị trí coherence được dẫn trong giao thức; khi bản khóa đã tồn tại, lệnh chỉ đối chiếu và giữ nguyên thời điểm khóa.

Verifier kiểm tra cả [bản khóa benchmark](benchmark/freeze.json) lẫn bản khóa cấu hình, nguồn, metric, đủ 150 xác nhận audit và kết quả chạy lại. Phần đính chính `q28` được tính lại từ trace và vị trí bằng chứng, rồi đối chiếu bảng đã lưu trong `human_review.json`. [Biên bản kiểm tra](results/artifact_review.json)

## Kết quả chính

| Cấu hình | Chunk | Hit@3 dev gốc | Hit@3 dev sau thẩm định | Hit@3 test | Tuân thủ filter test |
|---|---:|---:|---:|---:|---:|
| A · 320 token, chồng 48 | 162 | 17/20 | 18/20 | 6/10 | 1/2 |
| B · theo đoạn, tối đa 320 token | 156 | 18/20 | 19/20 | 5/10 | 1/2 |
| C · dùng B và bật filter khi có scope | 156 | 18/20 | 19/20 | 5/10 | 2/2 |

B được chọn trên kết quả dev gốc theo Hit@3, Hit@1 và p95 search, trước khi chạy test. Lần review sau thí nghiệm phát hiện `q28` có bằng chứng tương đương đủ trả lời ở top-1 của cả ba cấu hình nhưng chưa được gold gốc công nhận. Bảng sau thẩm định sửa cách tính câu này; thứ hạng, cấu hình, kết quả test và lựa chọn B giữ nguyên. [Bằng chứng, người review và bảng điểm điều chỉnh](benchmark/human_review.json) · [Báo cáo cá nhân](report/individual/2A202604113_PhamTranQuoc.md)

## Trạng thái xác nhận của học viên

**Phạm Trấn Quốc đã hoàn tất xác nhận:** 30 câu hỏi/gold, 30 mẫu coherence, 30 mẫu grounding và 90 kết quả metadata. [Hồ sơ xác nhận hiện tại](benchmark/human_review.json) ghi đúng tên, thời điểm, phạm vi và checksum bộ câu hỏi đã review. Nhật ký A/B/C đều ghi `human_audit_confirmed=True`; các lý do và nhãn từng mẫu nằm trong [audit](results/audits/).

`questions.jsonl` và `evidence_validation.json` là bản lịch sử đã khóa trước thí nghiệm. Những cờ `human_reviewed=false` hoặc dòng `human review pending` trong hai bản đó chỉ mô tả thời điểm ban đầu, không phải việc còn chờ học viên thực hiện. Xác nhận sau đó được lưu riêng trong `human_review.json` để giữ nguyên checksum. Phần thẩm định bằng chứng tương đương của Codex được công bố riêng, không gán thành một lần chấm thủ công mới của học viên.

## Nguồn

Corpus gồm tài liệu CPython v3.12.10 của Python Software Foundation và các tác giả đóng góp. Repo giữ RST gốc, 40 đơn vị nguồn đã chuẩn hóa, manifest và thông báo giấy phép. E5 dùng bản công khai có giấy phép MIT và tải riêng theo manifest. [Giấy phép corpus](data/licenses/) · [Đề và cấu trúc nộp](https://github.com/VinUni-AI20k/K5-L3-Day06-DataIngestionEngineering/blob/42f64eb230289ff5042abc03683252649dfbef79/SUBMISSION.md) · [Model E5](https://huggingface.co/intfloat/multilingual-e5-small)
