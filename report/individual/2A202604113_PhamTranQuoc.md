# Báo cáo cá nhân — Lab06 Data Ingestion Engineering

**Phạm Trấn Quốc · 2A202604113 · L3B**

## Dữ liệu và pipeline

Thí nghiệm so sánh ảnh hưởng của chia đoạn và lọc metadata đến khả năng tìm bằng chứng trong tài liệu Python 3.12.10. Corpus có **40 đơn vị nguồn từ tám tài liệu**, tổng 150.147 ký tự chuẩn hóa. Mỗi tài liệu đóng góp năm đơn vị đầu đủ điều kiện. Đơn vị ngắn nhất dài 3.600 ký tự, vượt mức tối thiểu 1.500 ký tự của đề đối với văn bản không phân trang. Trường `page` là số thứ tự đơn vị trong tài liệu. Manifest ghi cả phần được chọn và phần bị loại. [Manifest và nguồn](../../data/manifest.json) · [Giấy phép](../../data/licenses/)

Pipeline chuẩn hóa Unicode NFC và khoảng trắng, xử lý một phần cú pháp RST và giữ đoạn văn, ví dụ code cùng vị trí ký tự. Kết quả kiểm tra corpus ghi đủ 40 đơn vị và không có lỗi ingestion. Lượt tải model ban đầu gặp timeout, sau đó hoàn tất bằng tải từng khoảng byte và kiểm tra checksum của cùng revision. [Kiểm tra corpus](../../results/corpus_validation.json) · [Log lỗi ingestion](../../results/ingestion_errors.jsonl) · [Log môi trường](../../results/setup_events.jsonl)

Model local `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` tạo vector **384 chiều**. Query dùng tiền tố `query:`, đoạn nguồn dùng `passage:`. Qdrant client **1.20.0** lưu collection trên đĩa, chạy cosine exact search và hỗ trợ filter. Thử nạp lại cho số điểm không đổi, cập nhật loại ID cũ, xóa loại điểm hết hiệu lực và đóng mở giữ nguyên 162 điểm A, 156 điểm B. Phí API 0 USD. [Phiên bản và cấu hình chạy](../../results/run_manifest.json) · [Kiểm tra lưu trữ](../../results/storage_checks.json) · [Lệnh tái lập](../../README.md)

Mỗi chunk giữ `document_id`, `source`, `page` và offset để mở đúng đoạn trong nguồn. `document_type` lấy từ nhóm tài liệu chính thức `tutorial` hoặc `library`, dùng để giới hạn phạm vi khi câu hỏi nêu rõ. `section`, `language` và `version` bổ sung thông tin truy vết. Trong corpus này, version giống nhau ở mọi tài liệu nên không thu hẹp được tìm kiếm.

## Cấu hình và benchmark

| Cấu hình | Cách chia | Lọc metadata |
|---|---|---|
| A | Cửa sổ cố định 320 token, chồng 48 token | Tắt |
| B | Gom theo đoạn đến 320 token, không chồng lấn. Đoạn dài tách theo câu hoặc token | Tắt |
| C | Dùng cùng chunk, vector và collection với B | Bật nếu đầu vào nêu scope |

A–B thay cả ranh giới và mức chồng lấn. B–C chỉ thay filter. Cả ba dùng chung corpus, model, Qdrant, cosine và bộ câu hỏi. B/C trả cùng thứ tự khi câu hỏi không có filter. [Cấu hình](../../configs/strategies.json) · [Nhật ký và giả thuyết](../../artifacts/experiment_log.csv)

Benchmark tiếng Anh có **20 dev và 10 test**, gồm bốn câu filter dev và hai câu filter test. Mỗi câu có đáp án, đoạn evidence và nguồn/đơn vị. Kiểm tra trước khi chạy xác minh đoạn gold có trong nguồn chuẩn hóa và phù hợp filter. Query chỉ dùng câu hỏi, còn filter chỉ dùng scope đã nêu. Phép đo tự động gốc yêu cầu một chunk đúng nguồn và đơn vị, chứa đủ đoạn gold sau chuẩn hóa Unicode và khoảng trắng. Theo GUIDE, bằng chứng tương đương đủ trả lời cũng được chấp nhận khi có quyết định thẩm định được ghi lại; phần đính chính dưới đây áp dụng quy tắc này cho `q28`. [Câu hỏi và gold gốc](../../benchmark/questions.jsonl) · [Kiểm tra evidence trước thí nghiệm](../../benchmark/evidence_validation.json) · [Giao thức](../../benchmark/protocol.md)

Thứ tự chọn trên dev là Hit@3 cao, Hit@1 cao, p95 search thấp, rồi B/A/C nếu vẫn hòa. **B được chọn và khóa trước test.** [Lựa chọn](../../artifacts/selection.json) · [Khóa cấu hình](../../artifacts/config_freeze.json)

## Kết quả

Bảng dưới giữ nguyên kết quả tự động của lần chạy gốc. Các dòng dev là căn cứ chọn B trước test; các dòng test ghi kết quả đánh giá sau khi khóa cấu hình.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Chunk | p95 search | Phí API |
|---|---|---:|---:|---:|---:|---:|
| A | dev | 14/20 | 17/20 | 162 | 4,47 ms | 0 USD |
| A | test | 3/10 | 6/10 | 162 | 4,97 ms | 0 USD |
| B | dev | 17/20 | 18/20 | 156 | 2,97 ms | 0 USD |
| B | test | 3/10 | 5/10 | 156 | 3,53 ms | 0 USD |
| C | dev | 17/20 | 18/20 | 156 | 4,73 ms | 0 USD |
| C | test | 3/10 | 5/10 | 156 | 6,18 ms | 0 USD |

**Đính chính bằng chứng tương đương sau thí nghiệm.** `q28` hỏi ngoại lệ do ứng dụng định nghĩa nên kế thừa lớp nào thay cho `BaseException`. Top-1 của A/B/C đều có đoạn `library-exceptions`, đơn vị 2, nói rõ lớp do người dùng định nghĩa nên dùng `Exception`. Gold gốc bỏ sót đoạn này nên bộ so khớp tính trượt. Chunk A hạng 3 cũng giữ tên `Exception` và khuyến nghị đầy đủ, dù bị cắt một phần tiền tố tiêu đề. Codex đã đọc lại câu hỏi, đáp án, nguồn và từng chunk để công nhận hai đoạn tương đương; quote, offset, chunk bị ảnh hưởng, thời điểm và người review được lưu trong [hồ sơ thẩm định](../../benchmark/human_review.json).

| Cấu hình | Hit@1 dev sau thẩm định | Hit@3 dev sau thẩm định | Thay đổi so với phép đo gốc |
|---|---:|---:|---|
| A | 15/20 | 18/20 | Công nhận thêm `q28` ở top-1; hạng 3 cũng có bằng chứng hợp lệ |
| B | 18/20 | 19/20 | Công nhận thêm `q28` ở top-1 |
| C | 18/20 | 19/20 | Công nhận thêm `q28` ở top-1 |

Đây là sửa cách chấm bằng chứng sau thí nghiệm, không phải chạy lại một holdout mới. Câu hỏi, bộ chia, model, filter, thứ hạng và trace gốc không đổi. Điểm test của cả ba cấu hình không đổi; B vẫn là lựa chọn đã khóa trước test. Verifier tính lại bảng điều chỉnh từ trace và các quote đã công bố. `experiment_log.csv` giữ cột điểm gốc, đồng thời có các cột `_reviewed` để phân biệt rõ hai cách ghi nhận.

Hai trường hợp gần ranh giới được giữ nguyên: `q27` của A có thể đủ khi ghép hai chunk nhưng từng chunk chưa bao phủ toàn bộ đáp án đã khóa; `q26` của B/C có mã minh họa `bit_length` nhưng thiếu phát biểu bất đẳng thức và trường hợp số 0. Không dùng hợp nhiều chunk hoặc nới tiêu chí sau khi xem test để nâng điểm.

p95 chỉ đo truy vấn Qdrant. Embedding câu hỏi đo riêng, dùng lại vector cho A/B/C. A có 46.090 token, thời gian chia/embedding/lưu lần đầu lần lượt 0,357/11,565/1,174 giây. B có 40.218 token và mất 0,416/10,633/1,076 giây. C dùng lại index B. Chi phí phần cứng và điện chưa quy đổi. [Trace dev và test](../../results/traces/) · [Thời gian tìm kiếm](../../results/metrics/) · [Thống kê tài nguyên](../../results/resource_stats.json)

| Cấu hình | Coherence đã xác nhận /2 | Metadata đúng và đủ, đã xác nhận | Filter test tuân thủ | Grounding đã xác nhận 0/1/2 và trung bình /2 |
|---|---:|---:|---:|---|
| A | 1,10 (10 mẫu) | 30/30 | 1/2 | 3/1/6 · 1,30 |
| B | 1,50 (10 mẫu) | 30/30 | 1/2 | 3/2/5 · 1,20 |
| C | 1,50 (10 mẫu) | 30/30 | 2/2 | 4/1/5 · 1,10 |

**Nguồn nhãn và trạng thái hoàn tất.** Codex hỗ trợ code, benchmark và đề xuất nhãn coherence/grounding. Tôi, Phạm Trấn Quốc, đã đọc và xác nhận toàn bộ 30 câu hỏi/gold cùng 150 mục audit: 30 coherence, 30 grounding và 90 kết quả metadata. [Hồ sơ xác nhận hiện tại](../../benchmark/human_review.json) ghi phạm vi và thời điểm; [nhật ký A/B/C](../../artifacts/experiment_log.csv) đều đánh dấu đã xác nhận. Metadata được kiểm tra tự động với nguồn và offset trước khi tôi đối chiếu. Nhãn đề xuất, lý do gốc và xác nhận của tôi được giữ trong các trường riêng. [Hồ sơ audit](../../results/audits/)

Các cờ chưa review trong `questions.jsonl` và `evidence_validation.json` thuộc bản lịch sử trước thí nghiệm. Chúng không mô tả trạng thái công việc hiện tại. Tôi giữ nguyên các bản đã khóa và dùng manifest xác nhận riêng để bảo toàn checksum; phần thẩm định `q28` sau đó được ghi rõ là review do Codex hỗ trợ, không nhận là một lần chấm thủ công mới của tôi.

- **Coherence:** cùng 10 vị trí nguồn đã khóa cho mỗi cấu hình. Nếu nhiều chunk bao phủ vị trí, lấy chunk có điểm bắt đầu lớn nhất không vượt vị trí, rồi phá hòa theo ID. Chấm 0 nếu đoạn khó hiểu do thiếu ngữ cảnh, 1 nếu giữ được một phần ý, 2 nếu tự hiểu và đủ ý. B/C chia sẻ chunk nên dùng chung nhận xét về nội dung.
- **Grounding:** đọc cả top-3 của cùng 10 câu test mỗi cấu hình. Chấm 0 khi không có bằng chứng, 1 khi chỉ đủ một phần, 2 khi đủ để trả lời và truy nguồn. Mẫu đọc bỏ điểm và cờ hit để giảm ảnh hưởng của metric.
- **Metadata:** đối chiếu 90 kết quả với manifest, trang và offset. Tuân thủ filter tính theo câu có scope và đòi hỏi toàn bộ kết quả trả về nằm trong scope.

**Phạm vi xác nhận.** Mỗi cấu hình có 10 vị trí coherence, 10 câu grounding và 30 kết quả metadata đã xác nhận. Các trường `human_label`, `human_reason`, `human_reviewer` và `human_reviewed_at` ghi nhận đánh giá của tác giả. Để kiểm tra lại, dùng `position_id` và `chunk_id` cho coherence, `question_id` và `strategy` cho grounding, và thêm `rank` cho metadata. [Nhãn và metadata](../../results/audits/) · [Chunk](../../results/chunks/) · [Vị trí coherence](../../benchmark/coherence_positions.json) · [Trace test](../../results/traces/test.jsonl)

**Thiên lệch khi chấm.** Tôi là tác giả bài làm và cũng là người xác nhận nhãn, nên có thể ưu tiên cách diễn giải phù hợp với kỳ vọng của mình. Nhãn AI đề xuất trước đó cũng có thể ảnh hưởng nhận định của tôi. Mẫu grounding đã bỏ điểm similarity và cờ hit để giảm ảnh hưởng từ kết quả tự động, nhưng tôi vẫn biết chủ đề và đã xem nhãn đề xuất; đây chưa phải đánh giá độc lập hoàn toàn. Vì vậy, tôi giữ từng lý do cùng nguồn/chunk để người chấm có thể đối chiếu, công bố các trường hợp chưa đủ bằng chứng và không suy rộng từ mẫu nhỏ. Nếu mở rộng thí nghiệm, tôi sẽ nhờ một người khác chấm độc lập trước khi đối chiếu bất đồng.

## Phân tích và kết luận

**A–B:** B tăng một Hit@3 trên dev, tăng coherence trung bình từ 1,10 lên 1,50 và giảm khoảng 5.872 token lập chỉ mục. Trên test, A hơn B một câu. Ở `q26`, chunk A hạng 2 giữ định nghĩa `bit_length`, công thức và trường hợp số 0. Top-3 của B thiếu đoạn định nghĩa. Kích thước và ranh giới chunk ảnh hưởng cả ngữ cảnh giữ được lẫn thứ hạng.

**B–C:** C tăng tuân thủ phạm vi từ 1/2 lên 2/2 và giữ nguyên Hit@3 test. Ở `q29`, B trả một đoạn tutorial dù đề bài hỏi library. C lọc đúng library nhưng top-3 vẫn thiếu đoạn về `EOFError`. Metadata giúp kiểm soát phạm vi và truy nguồn, còn độ đủ của câu trả lời phụ thuộc vào đoạn được xếp hạng.

Hai ca trượt có thể mở lại trong trace test:

| Câu | Gold | Top-3 của B và nguyên nhân |
|---|---|---|
| `q14` | `library-json`, đơn vị 5 | Trả đơn vị 4/4/3, nhắc decoder và hook nhưng thiếu `raw_decode` cùng vị trí kết thúc JSON. |
| `q19` | `library-pathlib`, đơn vị 4 | Hạng 1 đúng đơn vị nhưng mô tả `is_relative_to` và `is_reserved`. Top-3 thiếu điều kiện đường dẫn tuyệt đối và `ValueError` của `as_uri`. |

**Kết luận:** giữ B theo tiêu chí dev đã chốt. B ưu tiên đoạn mạch lạc và dùng ít token hơn A. Khi phạm vi tutorial/library là yêu cầu bắt buộc, C có lợi thế vì áp dụng scope ngay lúc tìm kiếm. Tập test chỉ có 10 câu và dùng chung miền tài liệu với dev. Một câu tương ứng 10 điểm phần trăm, nên thứ hạng có thể đổi khi mở rộng benchmark. Chuẩn hóa RST còn sót markup, cách khớp trọn gold có thể bỏ sót diễn đạt tương đương và thời gian chỉ phản ánh máy hiện tại. Nếu làm lại, thử giữ tên phương thức cùng phần mô tả rồi đo trên một tập câu hỏi mới.

Quy trình tái lập dùng database và output riêng, đối chiếu 90 lượt truy xuất, thứ hạng và chỉ số gốc. Verifier còn kiểm tra cả hai bản khóa, đủ 150 xác nhận, trạng thái nhật ký A/B/C và bảng đính chính `q28`. Quy tắc chọn vị trí coherence nằm trong [build_benchmark.py](../../benchmark/build_benchmark.py). [Lệnh chạy](../../README.md) · [Kiểm chứng tái lập](../../results/artifact_review.json)
