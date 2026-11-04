# Hướng Nghiên Cứu Khoa Học Ứng Dụng Cho Hệ Thống eKYC

Tài liệu này đề xuất cách viết dự án eKYC CCCD dưới dạng **luận văn nghiên cứu
khoa học ứng dụng**. Nội dung gồm: định hướng đề tài, câu hỏi nghiên cứu, cấu
trúc luận văn, phương pháp đánh giá và checklist thực tế. Hãy đọc và quyết định
phần nào nên áp dụng.

---

## 1. Định Vị: Loại Nghiên Cứu Nào Phù Hợp?

| Loại | Yêu cầu | Mức độ phù hợp với dự án |
| --- | --- | --- |
| Nghiên cứu cơ bản / lý thuyết | Thuật toán, định lý hoặc kiến trúc mô hình mới, có chứng minh | Yếu — dự án đang dùng lại UltraFace, ArcFace, MiniFASNetV2, RapidOCR |
| **Nghiên cứu ứng dụng** | Giải một bài toán thực tế bằng cách tích hợp các phương pháp đã biết, kèm **bằng chứng định lượng** rằng giải pháp hiệu quả | **Rất phù hợp — đây là hướng đúng** |
| Báo cáo kỹ thuật thuần túy | Chỉ mô tả những gì đã xây dựng | Quá yếu nếu đứng một mình — cần chương đánh giá để trở thành nghiên cứu |

**Định hướng đề xuất**: "Thiết kế và đánh giá quy trình xác thực eKYC đa yếu tố
cho thẻ CCCD Việt Nam." Đóng góp khoa học không phải là một mạng nơ-ron mới,
mà là:

1. **Thiết kế hợp nhất đa yếu tố** (OCR + đối chiếu QR + so khớp khuôn mặt +
   liveness thụ động + liveness chủ động + phát hiện chỉnh sửa ảnh) áp dụng cho
   một loại giấy tờ đặc thù của Việt Nam — điều mà hầu hết các công bố eKYC hiện
   nay chưa phủ tới.
2. **Đặc trưng hóa bằng số liệu** từng module và của toàn hệ thống: độ chính
   xác, đánh đổi lỗi (FAR/FRR), khả năng chống tấn công giả mạo, độ trễ.
3. **Phân tích ablation** cho thấy yếu tố nào thực sự quan trọng.

---

## 2. Câu Hỏi Nghiên Cứu

Trình bày luận văn xung quanh 3–5 câu hỏi có thể trả lời được, ví dụ:

- **CH1 (OCR)**: RapidOCR/PaddleOCR trích xuất số hiệu và họ tên từ ảnh
  CCCD/CMND chụp trong điều kiện thực tế (mờ, nghiêng, ánh sáng khác nhau)
  chính xác đến mức nào?
- **CH2 (Khuôn mặt)**: Ngưỡng cosine similarity nào của embedding ArcFace cho
  đánh đổi FAR/FRR tốt nhất trên dữ liệu thu thập được, và ngưỡng đang dùng
  (`AI_MATCH_SCORE=0.30`) thể hiện ra sao?
- **CH3 (Chống giả mạo)**: Kết hợp liveness thụ động (MiniFASNetV2) và liveness
  chủ động (thách thức quay đầu) phát hiện tốt đến đâu các tấn công in giấy,
  chiếu màn hình và ảnh tĩnh?
- **CH4 (Hợp nhất)**: Việc thêm đối chiếu QR và phát hiện chỉnh sửa có làm giảm
  tỉ lệ nhận diện sai (false accept) so với chỉ dùng OCR + khuôn mặt không?
  (ablation)
- **CH5 (Hệ thống)**: Độ chính xác và độ trễ đầu-cuối của toàn pipeline là bao
  nhiêu, và khâu nào chiếm nhiều thời gian nhất?

Mỗi câu hỏi tương ứng với một thí nghiệm đo lường được — đây chính là điểm phân
biệt giữa luận văn nghiên cứu và báo cáo đồ án.

---

## 3. Phát Biểu Đóng Góp (Bản Nháp)

> Luận văn trình bày một pipeline eKYC hoàn chỉnh cho thẻ căn cước công dân Việt
> Nam. Pipeline kết hợp trích xuất trường dữ liệu bằng OCR, đối chiếu mã QR,
> phát hiện và so khớp khuôn mặt dùng ArcFace, liveness thụ động (MiniFASNetV2)
> và liveness chủ động (thách thức tư thế đầu), cùng bộ phát hiện ảnh bị đánh
> tráo, tất cả hợp nhất thành một quyết định duy nhất. Hệ thống được triển khai
> theo kiến trúc ba tầng (Next.js, NestJS, FastAPI). Từng module và hệ thống hợp
> nhất được đánh giá trên một bộ dữ liệu tự xây dựng bằng các chỉ số sinh trắc
> học chuẩn (FAR, FRR, APCER, BPCER), kèm phân tích ablation định lượng đóng góp
> của từng yếu tố.

Những điều **không nên** tuyên bố: một thuật toán mới, QR là bằng chứng mật mã,
hoặc hệ thống đạt mức bảo mật production. Hãy nêu rõ các giới hạn này trong phần
hạn chế.

---

## 4. Cấu Trúc Luận Văn Đề Xuất

Cấu trúc điển hình của một luận văn nghiên cứu ứng dụng, đối chiếu với những gì đã
có trong repository:

### Chương 1 — Mở đầu
- Bối cảnh: nhu cầu định danh từ xa; Quyết định 2345/QĐ-NHNN của Ngân hàng Nhà
  nước yêu cầu eKYC cho giao dịch trực tuyến; gian lận bằng thẻ in lại / phát
  lại màn hình.
- Đặt vấn đề, phạm vi (chỉ CCCD/CMND), câu hỏi nghiên cứu (§2), đóng góp dự kiến
  (§3), cấu trúc luận văn.

### Chương 2 — Cơ sở lý thuyết & tổng quan nghiên cứu
- OCR cho giấy tờ tùy thân (cổ điển + deep learning; nền tảng RapidOCR/PaddleOCR).
- Phát hiện / nhận diện khuôn mặt: UltraFace, landmark MediaPipe, embedding
  ArcFace, cosine similarity; định nghĩa FAR/FRR/EER.
- Chống giả mạo khuôn mặt: dòng MiniFASNet/V2; liveness thụ động vs chủ động;
  APCER/BPCER (ISO/IEC 30107-3).
- Cấu trúc mã QR trên CCCD Việt Nam; lý do QR là tín hiệu chống giả mạo chứ
  **không phải** bằng chứng mật mã.
- Tổng quan nghiên cứu liên quan: eKYC thương mại tại Việt Nam (FPT eKYC,
  Viettel, NganLuong, VNPT), các pipeline eKYC trong tài liệu học thuật; chỉ ra
  khoảng trống: rất ít công bố đánh giá định lượng một pipeline đa yếu tố hoàn
  chỉnh cho giấy tờ Việt Nam.

### Chương 3 — Thiết kế & triển khai hệ thống
Chương này hiện đã có sẵn khoảng 80% tư liệu trong repo:
- Kiến trúc: `Next.js → NestJS → FastAPI → PostgreSQL`
  ([README.md](README.md), [plan.md](plan.md)).
- Luồng xác thực và logic quyết định
  ([verify_service.py](ai/app/services/verify_service.py)): MATCH + passive
  live + active PASS + không bị reject QR + không bị reject tamper + đúng một
  khuôn mặt.
- Triển khai từng module:
  - OCR: RapidOCR là chính / PaddleOCR dự phòng, xoay theo EXIF, heuristic nhận
    số hiệu 12/9 chữ số ([ocr_service.py](ai/app/services/ocr_service.py)).
  - QR: bộ giải mã OpenCV + pyzbar, nhiều biến thể tiền xử lý, parser chịu thiếu
    trường, verdict `NOT_FOUND/MATCH/PARTIAL/MISMATCH`
    ([qr_service.py](ai/app/services/qr_service.py)).
  - Face: UltraFace 480×640, 5 landmark MediaPipe, alignment ArcFace 112×112,
    quét tile để tìm mặt in kích thước nhỏ trên thẻ
    ([face_service.py](ai/app/services/face_service.py)).
  - Liveness thụ động: crop 80×80 MiniFASNetV2, scale 2.7, TTA, dự phòng bằng
    Laplacian ([fasnet.py](ai/app/services/fasnet.py),
    [liveness_service.py](ai/app/services/liveness_service.py)).
  - Liveness chủ động: thách thức tư thế (thẳng / trái / phải), phát hiện frame
    tĩnh ([active_liveness_service.py](ai/app/services/active_liveness_service.py)).
  - Phát hiện chỉnh sửa: bộ phát hiện đánh tráo ảnh/thẻ
    ([tamper_net.py](ai/app/services/tamper_net.py),
    [tamper_service.py](ai/app/services/tamper_service.py)).
- Thiết kế bảo mật: lưu file riêng theo user, kiểm tra quyền sở hữu, frontend
  không gọi thẳng AI.

### Chương 4 — Thực nghiệm & đánh giá (phần lõi khoa học)
Xem §5 bên dưới. Đây là khoảng trống lớn nhất cần bổ sung.

### Chương 5 — Kết luận & hướng phát triển
- Trả lời CH1–CH5, tổng hợp các ngưỡng đã chọn.
- Hạn chế: quy mô và độ đa dạng của dữ liệu, chưa phát hiện được mask 3D /
  replay video, QR không phải bằng chứng mật mã, độ trễ khi chạy CPU, mẫu tấn
  công in giấy / màn hình.
- Hướng phát triển: huấn luyện mô hình face-PAD trên dữ liệu tấn công Việt Nam,
  triển khai edge, fine-tune OCR cho font tiếng Việt, liên kết phát hiện CCCD
  trùng.

---

## 5. Phương Pháp Đánh Giá

### 5.1 Các bộ dữ liệu cần xây dựng (hoặc mở rộng)

Hiện repo chỉ có vài fixture
([ai/tests/fixtures](ai/tests/fixtures)). Bộ dữ liệu đánh giá cần, theo từng
phân nhóm:

| Phân nhóm | Nội dung | Quy mô mục tiêu (đề xuất) |
| --- | --- | --- |
| Bộ OCR | Ảnh CCCD/CMND mặt trước + mặt sau kèm nhãn ground-truth (số hiệu, họ tên, ngày sinh) | 50–100 thẻ, nhiều góc chụp / mức mờ |
| Bộ khuôn mặt | Cặp selfie ↔ ảnh in trên thẻ, cùng người và khác người | ≥30 cặp cùng người, ≥30 cặp khác người |
| Bộ giả mạo | Khuôn mặt thật + ảnh in + replay từ màn hình của cùng chủ thể | ≥20 chủ thể × 3 dạng |
| Bộ liveness chủ động | Chuỗi quay trái/phải đúng + ảnh tĩnh + clip không chuyển động | ≥20 đúng, ≥20 tấn công |
| Bộ đầu-cuối | Gói hoàn chỉnh (front, back, selfie, liveness) kèm kết quả mong đợi | 40–100 gói |

Cách để tăng độ tin cậy:
- Bộ dữ liệu công khai cho liveness thụ động: CASIA-FASD, MSU-MFSD (in/replay),
  ROSE-Youtu — chỉ dùng để **đánh giá**, không huấn luyện.
- Bộ dữ liệu công khai để kiểm tra so khớp khuôn mặt: giao thức LFW.
- Dữ liệu tổng hợp cho tamper/QR: ảnh thẻ chỉnh sửa số hiệu, QR in lại.

Lưu ý về quyền riêng tư: dữ liệu CCCD thật rất nhạy cảm. Hãy mô tả quy trình
ẩn danh hóa và hủy dữ liệu trong luận văn (điểm này cũng được hội đồng đánh giá
cao).

### 5.2 Chỉ số cho từng module

| Module | Chỉ số |
| --- | --- |
| OCR | Độ chính xác theo trường (số hiệu, họ tên, ngày sinh), Character Error Rate (CER) |
| So khớp khuôn mặt | Phân bố cosine score cho cặp cùng/khác người; quét ngưỡng → FAR, FRR, EER; so sánh với ngưỡng đang deploy 0.30/0.20 |
| Liveness thụ động | APCER (tỉ lệ lỗi phân loại tấn công), BPCER (tỉ lệ lỗi với mặt thật), độ chính xác trên phân nhóm in giấy / màn hình |
| Liveness chủ động | Tỉ lệ hoàn thành thách thức của người thật, tỉ lệ từ chối clip tĩnh / không làm theo |
| Đối chiếu QR | Tỉ lệ đọc được QR (số thẻ đọc được / tổng số), tỉ lệ phát hiện mismatch trên QR bị sửa |
| Phát hiện chỉnh sửa | Độ chính xác trên ảnh thẻ thật vs ảnh bị đánh tráo |
| Đầu-cuối | Độ chính xác tổng thể, số lần false-accept / false-reject, độ trễ từng khâu (ms), phân bố độ trễ |

### 5.3 Phân Tích Ablation

Chạy lại bộ dữ liệu đầu-cuối với từng module tắt đi (phần lớn đã có env flag):

- Full pipeline (baseline)
- Không đối chiếu QR (`AI_QR_ENABLED=false`)
- Không phát hiện tamper (`AI_TAMPER_ENABLED=false`)
- Không liveness thụ động (`AI_FAS_ENABLED=false`)
- Không liveness chủ động (không gửi `liveness[]`)

Báo cáo mức thay đổi của false-accept rate — đây chính là câu trả lời trực tiếp
cho CH4 và là bằng chứng "nghiên cứu" mạnh nhất mà codebase hiện tại có thể cung
cấp.

### 5.4 Phân tích độ nhạy ngưỡng

Quét các ngưỡng chính và vẽ đường trade-off:

- `AI_MATCH_SCORE` (0.20–0.50) → ROC / EER
- `AI_FAS_REAL_THRESHOLD` (0.3–0.7) → APCER so với BPCER
- Ngưỡng frame tĩnh `AI_ACTIVE_STATIC_DIFF` (0.5–3.0)

### 5.5 Phần code cần bổ sung

Tests hiện tại là test chức năng (verdict có đúng không). Để làm nghiên cứu cần
**script đo lường**, ví dụ:

```
ai/evaluation/
├── dataset/            # nhãn, manifest, cấu hình split
├── run_ocr_eval.py     # độ chính xác theo trường + CER
├── run_face_eval.py    # phân bố score, quét FAR/FRR/EER
├── run_fas_eval.py     # APCER/BPCER
├── run_ablation.py     # toggle env flag, chạy lại bộ đầu-cuối
└── report.py           # tổng hợp kết quả ra CSV/bảng LaTeX
```

Các script này chỉ đọc, không đổi trọng số mô hình — chỉ đánh giá, không huấn
luyện. Các script trong [training/](ai/training) (`calibrate_tamper.py`,
`prepare_fantasyid.py`) rất hữu ích để mô tả cách các ngưỡng được xác định.

---

## 6. Chiến Lược Viết & Rủi Ro Cần Tránh

**Nên:**
- Dành 60% công sức cho Chương 4 (thực nghiệm). Hội đồng phản biện luận văn ứng
  dụng chủ yếu hỏi "bạn biết nó hoạt động bằng cách nào?".
- Báo cáo cả kết quả âm tính (ví dụ: MiniFASNetV2 không phát hiện được replay
  video — đã ghi trong phần hạn chế của README). Một hạn chế được nhận diện và
  đo lường đúng thể hiện sự chín chắn về khoa học.
- Biện minh cho mọi hằng số (0.30, 0.20, 0.5, 25.0, scale 2.7) bằng bài báo gốc,
  script huấn luyện, hoặc bằng chính phép quét ngưỡng của bạn.
- Có ít nhất một bảng so sánh với 2–3 hệ thống / công bố eKYC liên quan.

**Không nên:**
- Tuyên bố tính mới cho các mô hình đi vay. Tính mới nằm ở **tích hợp và cách
  đánh giá nó trên giấy tờ Việt Nam**.
- Gọi QR là bằng chứng xác thực mật mã — nó chỉ là tín hiệu nhất quán dữ liệu.
- Chỉ đưa ra ảnh chụp màn hình kiểu "chạy được rồi". Mỗi nhận định cần một con số.

---

## 7. Timeline Đề Xuất (12 tuần, full-time)

| Tuần | Công việc |
| --- | --- |
| 1–2 | Viết nháp Chương 1–2; chốt bộ câu hỏi nghiên cứu; kế hoạch thu thập dữ liệu + ẩn danh hóa |
| 3–4 | Xây dựng bộ dữ liệu đánh giá; viết script đo lường (§5.5) |
| 5–6 | Chạy đánh giá từng module (OCR, face, FAS); thu số liệu đầu tiên |
| 7–8 | Ablation + quét ngưỡng + chạy đầu-cuối; profile độ trễ |
| 9–10 | Chương 3 (triển khai) + Chương 4 (kết quả, bảng, hình vẽ) |
| 11 | Chương 5, abstract, tài liệu tham khảo, định dạng |
| 12 | Vòng phản biện với giảng viên hướng dẫn; chuẩn bị bảo vệ |

---

## 8. Checklist Trước Khi Bảo Vệ

- [ ] Mỗi câu hỏi nghiên cứu được trả lời bằng một số liệu đo được, hoặc một kết
      quả âm tính rõ ràng
- [ ] Có FAR/FRR/EER cho so khớp khuôn mặt kèm lý do chọn ngưỡng
- [ ] Có APCER/BPCER cho liveness thụ động
- [ ] Có bảng ablation thể hiện tác động riêng của từng yếu tố
- [ ] Có phân tách độ trễ theo từng khâu
- [ ] Mô tả dữ liệu: kích thước, nguồn, nhãn, split, cách ẩn danh
- [ ] Mọi ngưỡng và hằng số đều được biện minh
- [ ] Phần hạn chế: mask 3D, replay video, QR không mật mã, độ trễ CPU, tính đại
      diện của dữ liệu
- [ ] Có ít nhất một so sánh với công trình công bố liên quan
- [ ] Có phần bảo mật / quyền riêng tư mô tả thiết kế lưu trữ và kiểm tra quyền
      sở hữu
