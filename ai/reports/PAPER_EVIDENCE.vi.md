# Bộ bằng chứng thực nghiệm cho bài báo — AI service (IDVerification)

> **Mục đích.** Gom toàn bộ số liệu *đã chạy thật* trên máy, kèm giao thức và
> lệnh tái lập, để viết Chương 4 (Kết quả) mà không phải nhớ lại quy trình.
> Mọi con số dưới đây đều đọc trực tiếp từ file JSON/CSV trong `ai/reports/`.
>
> **Nguyên tắc chốt số:** ngưỡng và trọng số chọn trên **val**, chỉ áp lên
> **test** một lần; không chọn lại trên test. Các lần chạy lại dưới đây tái
> lập đúng từng chữ số thập phân.
>
> Ngày chốt snapshot: 2026-10-22 · pytest: **44 passed** · dataset: FantasyID
> (`ai/data/fantasyid`) + CASIA-FASD crops (`ai/data/archive (1)`) + LFW
> deep-funneled (`ai/data/lfw-deepfunneled`) — đều gitignore.

---

## 1. Môi trường đo

| Thành phần | Giá trị |
|---|---|
| OS / shell | Windows 26H2 · PowerShell |
| Python | 3.11.9 (global) |
| PyTorch | 2.13.0 + cu126 · GPU RTX 4060 Laptop 8 GB |
| ONNX Runtime | CPUExecutionProvider (UltraFace, ArcFace) |
| OCR | RapidOCR (onnxruntime), ảnh gốc độ phân giải cao |
| Điều kiện chạy | `cd ai` + `$env:PYTHONIOENCODING="utf-8"` (bắt buộc để log tiếng Việt) |

---

## 2. Giao thức đánh giá (viết cho mục Methodology)

1. **Split.** Dùng đúng train/test chính thức của FantasyID. Tập **val** = 20%
   train, cắt **một lần** bằng `torch.Generator().manual_seed(42)` +
   `randperm`, đóng băng thành `models/fantasyid_val.txt` (**284 ảnh: 116 bona
   fide / 168 forged**). Mọi lần chạy lại dùng đúng manifest này.
2. **Kỷ luật ngưỡng.** Calibration (`training/calibrate_tamper.py`) chỉ thấy
   val để chọn `AI_TAMPER_THRESHOLD` và `model_weight`. Evaluation
   (`evaluation/run_tamper_eval.py`) nhận ngưỡng từ calibration rồi chấm test;
   báo cả 3 ngưỡng (0.55 triển khai, 0.57 best-F1 val, 0.675 FPR-10% val).
3. **Chỉ số.**
   - Tamper: F1/Precision/Recall/FPR/FNR tại ngưỡng cố định; **AUC** (Mann–Whitney,
     xử lý hoà điểm) và **AP** (không phụ thuộc ngưỡng); recall theo loại tấn
     công; **IoU** định vị vùng sửa so với mask GT.
   - FAS (ISO/IEC 30107-3): **APCER** (tấn công bị nhận là thật), **BPCER**
     (người thật bị từ chối), **ACER**, **EER**.
   - Khớp mặt: FAR/FRR/EER trên cặp genuine/impostor.
   - Độ trễ: median/p95 mỗi module + tổng tuần tự.
4. **Chống rò rỉ tên file.** `prepare_fantasyid.py` đặt tên
   `<device>__<stem>.jpg` cho **mọi** file (không chỉ khi trùng stem) — bản cũ
   để lại nhóm "(no-device)" trùng 100% forged ở test, đã sửa và tái lập đúng
   số cũ (xem `reports/tamper-fantasyid-findings.vi.md` §7).
5. **Ablation không suy luận lại.** Điểm từng ảnh được dump
   (`--scores-csv`); `score(w) = w*model + (1-w)*signal` là tuyến tính nên
   quét lại mọi `w` bằng phép biến đổi, không chạy mạng lại.
6. **FAS.** Gói CASIA-FASD crops có split lỗi (20/20 subject train nằm trong
   test; 1.403 frame trùng khoá), nên gộp + dedup thành 2.660 frame duy nhất
   (674 thật / 1.986 giả, 30 người / 360 video) rồi chấm **một lần**; không
   huấn luyện trên bộ này. Chấm trực tiếp trên crop (`--no-detect`) để không
   trộn lỗi detector; đơn vị chính là **video** (trung bình điểm các frame),
   frame-level báo kèm. Ngưỡng giữ nguyên 0,5 như triển khai, không chọn lại.
7. **Độ bền FAS.** Giữ checkpoint + ngưỡng 0,5, TTA tắt; áp 8 phép suy giảm
   tổng hợp lên đúng 2.660 crop (JPEG q70/50/30, Gaussian blur σ=1,5/3,0,
   hạ–nâng 128/64 px, giảm sáng 0,5×), mỗi ảnh đọc một lần rồi chấm lại từng
   điều kiện. Không huấn luyện lại, không chọn ngưỡng trên tập này; `clean`
   tái lập đúng §6.4 (video EER 0,0222; AUC 0,9981).
8. **Khớp mặt LFW.** LFW View 1 deep-funneled: 1.100 + 1.100 cặp DevTrain
   (val) và 500 + 500 cặp DevTest (test). Trích embedding bằng đúng
   `FaceService` (UltraFace → MediaPipe 5 điểm → ArcFace 112×112), cache ảnh;
   điểm cặp = cosine. Chọn ngưỡng EER trên val, báo test một lần; báo thêm
   ngưỡng production 0,30 và TAR@FAR. Cặp no-face nhận score 0; báo cáo
   `detected_only` để tách lỗi detect khỏi chất lượng nhận dạng.

---

## 3. Kết quả module tamper (Real)

### 3.1 Calibration trên val (chọn ngưỡng)

| Chỉ số | Giá trị |
|---|---|
| Best F1 | **0.8113 @ thr = 0.570** (P = 0.7701 · R = 0.8571) |
| FPR = 10% | thr = 0.675 → F1 = 0.7852 (P = 0.900 · R = 0.696) |
| Model-only @0.5 | F1 = 0.8000 (P = 0.710 · R = 0.917) |
| Gợi ý triển khai | `AI_TAMPER_THRESHOLD=0.57`, `AI_TAMPER_SUSPECT=0.42` |

Nguồn: `models/tamper_calibration.json` · lệnh: `training/calibrate_tamper.py`.

### 3.2 Test held-out (1.385 ảnh: 300 bona / 1.085 forged)

| Ngưỡng | F1 | Precision | Recall | FPR | FNR |
|---|---|---|---|---|---|
| **0.55 (đang triển khai)** | **0.4481** | 0.6710 | 0.3364 | 0.5967 | 0.6636 |
| 0.57 (best-F1 val) | 0.4301 | 0.6603 | 0.3189 | 0.5933 | 0.6811 |
| 0.675 (FPR-10% val) | 0.2839 | 0.6278 | 0.1834 | 0.3933 | 0.8166 |

| Chỉ số phi ngưỡng | Giá trị |
|---|---|
| AUC | **0.3417** (model-only: 0.3190) |
| Average Precision | **0.7429** |
| Recall — tấn công `face` (n = 150) | 0.7933 |
| Recall — tấn công `text` (n = 935) | **0.2631** |
| IoU định vị (n = 911 mask GT) | 0.0 / 0.0 (mask head không phát vùng nào) |
| Độ trễ module (median / p95) | 252.4 / 461.0 ms |

Nguồn: `reports/tamper_test.json`, `reports/tamper_test_scores.csv`.
Diễn giải (đã phân tích trong findings): val → test sụp vì **domain shift +
device shortcut** (train chỉ có `face_text` + 3 thiết bị; test là OOD
`digital_3`/`facedancer`/`textdiffuserft_bfei`, bona fide test **toàn bộ là
`iphone15`** — thiết bị không có trong train), cộng thêm tấn công văn bản khó
nhất. AUC < 0.5 là hệ quả trực tiếp của đảo điểm theo thiết bị: trong từng
thiết bị hai lớp, bona ≥ forged (bảng dưới), nên xếp hạng toàn cục bị đảo.

### 3.3 Điểm theo thiết bị (test, thr = 0.55)

| Thiết bị | Nhãn | n | tamper_score TB | model_prob TB | % Dương tính |
|---|---|---|---|---|---|
| huawei | bona | 100 | 0.588 | 0.821 | 80.0% |
| huawei | forged | 361 | 0.405 | 0.553 | 45.7% |
| iphone15 | bona | 100 | 0.444 | 0.627 | 56.0% |
| iphone15 | forged | 100 | 0.482 | 0.655 | 62.0% |
| scan | bona | 100 | 0.374 | 0.527 | 43.0% |
| scan | forged | 362 | 0.225 | 0.308 | 24.9% |
| iphone15pro | forged | 262 | 0.173 | 0.242 | 18.3% |

Nguồn: `evaluation/analyze_tamper_scores.py`. Khoảng cách **giữa thiết bị**
lớn hơn khoảng cách **giữa hai lớp** — bằng chứng định lượng cho shortcut
theo thiết bị.

### 3.4 Ablation trọng số fusion (ngưỡng chọn lại trên val cho từng w)

`score(w) = w*model + (1-w)*signal`, signal = 4 tín hiệu heuristic.

| w_model | val AUC | val F1 @ thr | test AUC | test AP | test F1 @0.55 | FPR @thr |
|---|---|---|---|---|---|---|
| 0.0 (heuristic-only) | 0.9603 | 0.8861 @ 0.050 | **0.5172** | 0.8338 | 0.0000 | 0.1433 |
| 0.3 | 0.9111 | 0.8713 @ 0.310 | 0.3619 | 0.7624 | 0.0000 | 0.2800 |
| 0.5 | 0.8822 | 0.8247 @ 0.495 | 0.3521 | 0.7541 | 0.1081 | 0.3500 |
| **0.7 (triển khai)** | 0.8619 | 0.8146 @ 0.565 | 0.3417 | 0.7430 | 0.4481 | 0.5967 |
| 1.0 (model-only) | 0.8112 | 0.8057 @ 0.815 | 0.3190 | 0.6989 | **0.5070** | 0.5933 |

Nguồn: `reports/tamper_ablation.json` · lệnh:
`python evaluation/ablate_tamper_weights.py --val-csv reports/tamper_val_scores.csv --test-csv reports/tamper_test_scores.csv`.

Ba kết luận cho bài báo:

1. **Heuristic học "dấu vân tay thiết bị"**: AUC 0.9603 trên val nhưng về
   ngẫu nhiên (0.5172) trên test OOD → không dùng làm tín hiệu quyết định.
2. **Model là nguồn tín hiệu chuyển miền tốt hơn**: AUC/AP cao hơn ở mọi w < 1;
   nhưng tại ngưỡng triển khai 0.55, model-only cho F1 cao hơn (0.5070 so với
   0.4481) → nếu buộc phải chọn một cấu hình vận hành, **w = 1.0 + ngưỡng
   0.55** tốt hơn; giữ fusion 0.7 chỉ để giữ khả năng giải thích.
3. Heuristic chỉ nên là **thông tin phụ (advisory)**, không tham gia chấm điểm
   quyết định — đúng với khuyến nghị "chuyển tamper sang advisory".

> Sai lệch nhỏ: ablation đọc CSV đã làm tròn 4 chữ số nên val F1 @w=0.7 =
> 0.8146 (thr 0.565) thay vì 0.8113 (thr 0.570) của calibration. Số **chính
> thức** lấy từ `models/tamper_calibration.json`.

---

## 4. Độ trễ & độ phủ phát hiện (Real)

Đo 200 ảnh test lấy đều, warm-up trước khi tính; tamper lấy từ 1.385 ảnh.

| Module | n | median (ms) | p95 (ms) | Ghi chú |
|---|---|---|---|---|
| io (imread + RGB) | 400 | 12.9 | 51.4 | 2 phép/ảnh |
| detect (UltraFace) | 200 | 33.0 | 133.6 | ảnh gốc độ phân giải cao |
| landmark (MediaPipe) | 200 | 15.8 | 35.3 | |
| embed (ArcFace) | 110 | 6.8 | 67.2 | chỉ 110/200 ảnh có đủ 5 điểm |
| fas (MiniFASNetV2 + TTA) | 113 | 185.3 | 241.8 | TTA = 2 forward |
| ocr (RapidOCR) | 200 | **2.329,8** | 3.381,4 | **80,6% tổng thời gian** |
| qr (fixture CCCD) | 20 | 56.5 | 63.1 | 1 ảnh, lặp 20 lần |
| tamper (TamperNet) | 1.385 | 252.4 | 461.0 | từ `tamper_test_scores.csv` |
| **Tổng tuần tự (median)** | | **2.892,4** | | chưa gồm upload mạng |

Độ phủ phát hiện trên ảnh chân dung in trên thẻ: **113/200** ảnh có box
UltraFace, **110/200** có landmark, **87/200 (43,5%) không phát hiện được
mặt** ở độ phân giải gốc. Đây là lỗ hổng thật của nhánh "mặt trên thẻ", cần
tiling khi detect (hàm `_tile_images` đã có nhưng chỉ dùng trong fallback của
`_aligned_embedding`).

Nguồn: `reports/latency.json` · lệnh:
`python evaluation/bench_pipeline.py --data data/fantasyid/prepared/test --limit 200 --out reports/latency.json`.

**Hàm ý kỹ thuật (đóng góp thực tiễn):**
- OCR chiếm ~80% độ trễ → hạ cạnh dài về ~1.600 px trước khi OCR, crop vùng
  thẻ, hoặc chạy OCR song song với các module khác sẽ đưa tổng xuống ~1 s.
- FAS 185 ms ở ảnh gốc (có detect) do TTA; trên crop 256×256 chỉ đo classifier:
  17,0 ms (TTA) / 8,5 ms (không TTA) — tắt TTA giảm ~50% và **không làm xấu**
  EER cấp video (2,22% vs 3,33%; xem §6.4).

---

## 5. Kiểm thử đơn vị (Real)

```
cd ai; $env:PYTHONIOENCODING="utf-8"; python -m pytest tests -q
→ 44 passed in 3.67s
```

Bao phủ: OCR parse số giấy tờ, QR decode/parse/compare (fixture
`tests/fixtures/cccd_back_qr.png`), face service (align/embed/cosine),
liveness (MiniFASNet + fallback blur + blur gate `LOW_QUALITY`/`RETRY`),
face fallback landmark → detect box, tamper (fusion, ngưỡng, mask regions),
active liveness. Đây là "regression suite" trích dẫn được trong mục
*Implementation & Testing*.

---

## 6. FAS trên CASIA-FASD (Real)

### 6.1 Dữ liệu và giao thức

- **Nguồn:** gói CASIA-FASD bản **crop khuôn mặt** ở `ai/data/archive (1)`
  (4.063 ảnh JPEG 256×256; tên `<subject>_<HR?>_<video>.avi_<frame>_<real|fake>.jpg`).
  30 người / 360 video; nhãn chỉ `real`/`fake`, **không tách print/replay**.
- **Split của gói không dùng được:** 20/20 subject của `train_img` nằm trong
  `test_img` và **1.403 frame trùng khoá** `(subject, quality, video, frame)`.
  Vì ta không huấn luyện trên bộ này, ta gộp hai split rồi dedup → **2.660 frame
  duy nhất (674 thật / 1.986 giả)**; trong bài **không gọi** đây là split test
  chính thức.
- **Chấm:** `MiniFASNetV2` trực tiếp trên crop (`--no-detect`), ngưỡng 0,5
  (`AI_FAS_REAL_THRESHOLD`) giữ nguyên như triển khai — không chọn lại trên
  dữ liệu này. Đơn vị chính là **video** (trung bình `real_prob` các frame) vì
  các frame trong cùng clip tương quan mạnh; frame-level báo kèm để so sánh.
- **Robustness (§6.5):** dùng đúng 2.660 crop này, không thêm dữ liệu; mỗi
  điều kiện áp lên ảnh rồi chấm lại với cùng ngưỡng 0,5 và TTA tắt.
- **Tái lập:**
  ```powershell
  cd ai
  $env:PYTHONIOENCODING="utf-8"
  python evaluation/prepare_casia_crops.py --src "data/archive (1)" --out data/fas_casia
  python evaluation/run_fas_eval.py --real data/fas_casia/real --attack data/fas_casia/attack `
      --no-detect --tag casia_fasd_crops `
      --scores-csv reports/fas_casia_crops_scores.csv --out reports/fas_casia_crops.json
  python evaluation/analyze_fas_scores.py --scores-csv reports/fas_casia_crops_scores.csv `
      --manifest data/fas_casia/manifest.csv --out reports/fas_casia_crops_video.json
  ```

### 6.2 Kết quả chính (TTA bật — cấu hình triển khai)

| Mức | n (thật/tấn công) | APCER | BPCER | ACER | EER | AUC | median real_prob (thật/tấn công) |
|---|---|---|---|---|---|---|---|
| Frame | 674 / 1.986 | 0,0136 | 0,1231 | 0,0684 | 0,0639 @0,176 | 0,9874 | 0,9778 / 0,0011 |
| **Video** | 90 / 270 | **0,0000** | **0,1000** | **0,0500** | **0,0333 @0,264** | **0,9972** | 0,9475 / 0,0039 |

Đường vận hành (frame, chọn theo BPCER):

| Mục tiêu | Ngưỡng | APCER | BPCER |
|---|---|---|---|
| BPCER = 5% | 0,122 | 0,0856 | 0,0504 |
| BPCER = 10% | 0,360 | 0,0252 | 0,1009 |

### 6.3 Theo chất lượng video (TTA bật, ngưỡng 0,5)

| Nhóm | n (thật/tấn công) | APCER | BPCER | ACER | EER |
|---|---|---|---|---|---|
| `HR` | 30 / 90 | 0,0000 | **0,2333** | 0,1167 | 0,0389 @0,179 |
| `none` (LR) | 60 / 180 | 0,0000 | 0,0333 | 0,0167 | 0,0167 @0,401 |

BPCER nhóm HR = 7/30 video thật bị từ chối; cỡ mẫu nhỏ nên khoảng tin cậy
rộng — chưa đủ để kết luận "HR khó hơn LR".

### 6.4 Ablation TTA

| Cấu hình | Frame EER | Video BPCER @0,5 | Video EER | Video AUC | Latency median / p95 |
|---|---|---|---|---|---|
| TTA bật (mặc định) | **0,0639** | 0,1000 | 0,0333 @0,264 | 0,9972 | 16,96 / 17,59 ms |
| TTA tắt (`AI_FAS_TTA=false`) | 0,0669 | **0,0778** | **0,0222 @0,303** | **0,9981** | **8,48 / 8,84 ms** |

TTA **không** mang lại lợi ích rõ ràng: frame EER tốt hơn 0,3 điểm phần trăm
nhưng video EER lại xấu hơn 1,1 điểm phần trăm (chênh 1 video thật trên 90) và
độ trễ gấp đôi. Khuyến nghị: **tắt TTA** cho triển khai mặc định; giữ TTA nếu
cần điểm ổn định hơn trên từng ảnh đơn lẻ.

### 6.5 Độ bền trước suy giảm chất lượng ảnh (TTA tắt)

Ảnh selfie/CCCD thực tế thường bị nén JPEG, mờ do rung tay, hạ độ phân giải
khi gửi qua mạng hoặc thiếu sáng. Giữ nguyên checkpoint và ngưỡng 0,5, TTA
tắt (§6.4), áp từng điều kiện lên **cả 2.660 crop** rồi chấm lại — không
huấn luyện lại, không chọn ngưỡng trên tập này. Điều kiện `clean` tái lập
đúng baseline TTA-off: video APCER 0, BPCER 7,78%, EER 2,22% @0,303,
AUC 0,9981.

| Điều kiện | APCER | BPCER | ACER | EER (video) | AUC | ΔEER vs clean |
|---|---|---|---|---|---|---|
| `clean` | 0,0000 | 0,0778 | 0,0389 | 0,0222 @0,303 | 0,9981 | — |
| JPEG q70 | 0,0000 | 0,1000 | 0,0500 | 0,0333 @0,237 | 0,9977 | +0,0111 |
| JPEG q50 | 0,0074 | 0,1111 | 0,0593 | 0,0333 @0,233 | 0,9977 | +0,0111 |
| JPEG q30 | 0,0074 | 0,1111 | 0,0593 | 0,0444 @0,195 | 0,9968 | +0,0222 |
| Gaussian blur σ=1,5 | 0,0000 | 0,2333 | 0,1167 | 0,0778 @0,084 | 0,9780 | +0,0556 |
| Gaussian blur σ=3,0 | 0,0370 | **0,6222** | 0,3296 | **0,2889** @0,034 | 0,8271 | +0,2667 |
| Hạ–nâng 128 px | 0,0000 | 0,1333 | 0,0667 | 0,0648 @0,122 | 0,9900 | +0,0426 |
| Hạ–nâng 64 px | 0,0037 | 0,3556 | 0,1796 | 0,1222 @0,066 | 0,9478 | +0,1000 |
| Giảm sáng 0,5× | 0,0074 | 0,0778 | 0,0426 | 0,0241 @0,277 | 0,9976 | +0,0019 |

BPCER nhóm HR tệ hơn nhiều khi ảnh bị mờ/hạ phân giải (n = 30 video thật /
90 tấn công; `none` = 60/180):

| Điều kiện | BPCER `none` | BPCER `HR` | APCER `none` | APCER `HR` |
|---|---|---|---|---|
| `clean` | 3,33% | 16,67% | 0,00% | 0,00% |
| JPEG q30 | 5,00% | 23,33% | 1,11% | 0,00% |
| Blur σ=1,5 | 11,67% | 46,67% | 0,00% | 0,00% |
| Blur σ=3,0 | 48,33% | 90,00% | 5,00% | 1,11% |
| Hạ–nâng 128 | 3,33% | 33,33% | 0,00% | 0,00% |
| Hạ–nâng 64 | 15,00% | 76,67% | 0,56% | 0,00% |
| Giảm sáng 0,5× | 1,67% | 20,00% | 0,00% | 2,22% |

**Đọc kết quả:**

- **JPEG không phải nút thắt:** q30 vẫn giữ APCER 0,74%, BPCER 11,11%,
  EER 4,44% — chỉ tăng ~2,2 điểm phần trăm EER so với ảnh sạch.
- **Mờ là failure mode chính:** σ=1,5 làm BPCER 23,33% và EER 7,78%; σ=3,0
  làm BPCER 62,22%, EER 28,89%, APCER 3,70%, AUC rơi còn 0,8271. Nguyên nhân:
  MiniFASNet dựa vào texture tần số cao; làm mờ xóa dấu vân tay này nên mặt
  thật bị xếp nhầm sang print/replay (false spoof), không phải false accept.
- **Hạ phân giải cũng nguy hiểm:** 128 px → BPCER 13,33%, EER 6,48%; 64 px →
  BPCER 35,56%, EER 12,22%, dù APCER vẫn thấp (0,37%).
- **Thiếu sáng 0,5× gần như vô hại:** BPCER giữ 7,78%, APCER 0,74%, EER tăng
  nhẹ lên 2,41% — nên ưu tiên chống mờ/hạ phân giải hơn là tăng sáng.
- **Nhóm HR nhạy hơn:** blur σ=1,5 làm BPCER HR 46,67% (so với 11,67% nhóm
  `none`); hạ 64 px làm 76,67% (so với 15,00%). Nhóm HR có biên quyết định
  mỏng hơn ngay cả khi sạch (16,67% vs 3,33%), nên suy giảm đẩy qua ngưỡng.

**Hàm ý kỹ thuật:**

1. **Đã triển khai blur gate trước FAS** (`AI_FAS_BLUR_GATE`, mặc định 25;
   `0` để tắt): Laplacian variance đo trên crop mặt; dưới ngưỡng trả
   `LOW_QUALITY` + `retake_required=true`, không để FAS tự kết luận "spoof".
   Trung vị Laplacian variance của ảnh thật sạch là **133,8**; chỉ **1,3%**
   ảnh thật sạch dưới 25, trong khi **99,9%**
   ảnh blur σ=1,5 và **100%** ảnh blur σ=3,0 nằm dưới ngưỡng này.
   `verify_service` trả `RETRY` + `retake_reason` nếu không có lỗi giấy tờ/đa
   mặt; tamper/QR/mất mặt trên thẻ vẫn `REJECTED`. Ngưỡng 25 vẫn cần hiệu
   chỉnh trên ảnh điện thoại thật.
2. **Không hạ độ phân giải/khử nhiễu mạnh trước FAS**: giữ cạnh dài của crop
   ≥128 px và tránh Gaussian/denoise; nếu phải nén để gửi mạng, giữ JPEG ≥ q70.
3. Nếu fine-tune MiniFASNet, thêm augmentation blur/JPEG/downscale để model
   không học "mờ = giả"; nếu không, giữ nguyên checkpoint và xử lý bằng gate.

> **Đã triển khai (2026-10-22):** `LivenessService` trả `LOW_QUALITY` trước
> MiniFASNet; `VerifyService` trả `RETRY`; test `tests/test_liveness_gate.py`.
> Đây là thay đổi API: client cần xử lý thêm status `RETRY`.

### 6.6 Diễn giải và giới hạn

- Với TTA bật (§6.2), ở ngưỡng triển khai 0,5: **không có video tấn công nào
  qua** (APCER 0/270), 10% video thật bị từ chối; hạ về ngưỡng EER 0,264 đổi
  lại 3,33% APCER (9/270) để giảm BPCER còn 3,33% (3/90). Với TTA tắt
  (khuyến nghị §6.4), BPCER 7,78% và EER 2,22% (§6.5).
- MiniFASNet (tiền huấn luyện trên CelebA-Spoof) **chuyển miền tốt** sang
  CASIA-FASD, dù đây là bộ cũ, độ phân giải thấp.
- **Không suy rộng trực tiếp sang CCCD:** đầu vào là crop 256×256 đã căn mặt
  (bỏ qua detect/align), ảnh JPEG nén lại, và không có nhãn print/replay để
  tách APCER theo loại tấn công. Đây là đánh giá **classifier**, không phải
  đánh giá pipeline trên ảnh thẻ thật.
- Gói crops là bản chia lại không chính thức; đã dedup như §6.1. Muốn kết quả
  theo video gốc + tách print/replay thì cần bản đầy đủ ~2,0 GB (§8).
- **Robustness dùng suy giảm tổng hợp**, không phải pipeline camera thật:
  JPEG nén lại trên crop đã nén, blur Gaussian đều, hạ–nâng bằng nội suy;
  không mô phỏng nhiễu cảm biến, rung tay hay auto-exposure. Kết quả định
  hướng chất lượng, không phải cam kết SLA.
- **Nhóm HR nhỏ** (30 video thật/90 tấn công) nên BPCER 90% ở blur σ=3,0
  tương ứng 27/30 video, khoảng tin cậy vẫn rộng.

---

## 7. Face verification trên LFW (Real)

### 7.1 Dữ liệu và giao thức

- **LFW View 1, deep-funneled:** 13.233 ảnh / 5.749 người; ảnh đã căn mặt,
  khuôn mặt lớn, điều kiện dễ hơn ảnh CCCD/selfie thực tế.
- **Val (DevTrain):** 1.100 cặp genuine + 1.100 cặp impostor = 2.200 cặp.
- **Test (DevTest):** 500 + 500 = 1.000 cặp; chỉ chấm một lần.
- Mỗi ảnh được trích embedding bằng đúng `FaceService`: UltraFace
  `version-RFB-640.onnx` → MediaPipe 5 landmark → ArcFace `recognition.onnx`
  (112×112). Ảnh được cache theo đường dẫn: 4.992 ảnh unique, 1.408 cache hit,
  2 ảnh không detect được mặt.
- Điểm một cặp = cosine similarity. Ngưỡng EER chọn trên val, áp nguyên sang
  test; ngưỡng production `AI_MATCH_SCORE=0,30` báo cáo song song.
- Cặp có ảnh no-face nhận score 0 và được tính vào FAR/FRR chính; báo thêm
  `detected_only` để tách lỗi detect khỏi chất lượng embedding.

Lệnh tái lập (chạy từ `ai/`):

```powershell
$env:PYTHONIOENCODING="utf-8"
python evaluation/run_face_eval.py --root data/lfw-deepfunneled/lfw-deepfunneled `
  --pairs-dir data --out reports/face_lfw.json `
  --scores-csv reports/face_lfw_scores.csv
```

### 7.2 Kết quả

| Split | N cặp | FAR @0,30 | FRR @0,30 | Acc @0,30 | EER | EER thr | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (DevTrain) | 2.200 | 0,0000 | 0,0282 (31/1.100) | 0,9859 | 0,0145 | 0,1615 | 0,9933 |
| test (DevTest) | 1.000 | 0,0000 | 0,0340 (17/500) | 0,9830 | 0,0200 | 0,1600 | 0,9918 |

Tại ngưỡng EER chọn trên val (0,1615), test:

| FAR | FRR | TAR | Accuracy |
|---:|---:|---:|---:|
| 0,0160 (8/500) | 0,0200 (10/500) | 0,9800 | 0,9820 |

TAR@FAR (ngưỡng chọn trên val):

| FAR mục tiêu | Ngưỡng val | FAR test | TAR test |
|---:|---:|---:|---:|
| 1% | 0,1797 | 0,0160 | 0,9800 |
| 0,1% | 0,2298 | 0,0040 | 0,9760 |

- **Detected-only:** val 2.197 cặp (1.098 genuine / 1.099 impostor),
  EER 0,0137, AUC 0,9942; test 1.000 cặp, EER 0,0200, AUC 0,9918.
- **Phân bố điểm:** trung vị genuine 0,592 (val) / 0,606 (test); impostor
  0,0076 / 0,0092. Impostor cao nhất 0,2298 (val) / 0,2362 (test), dưới
  ngưỡng production 0,30.

### 7.3 Đọc kết quả

- Ở ngưỡng triển khai 0,30: **không có false accept** trên 500 cặp impostor
  test; 17/500 cặp genuine bị từ chối (FRR 3,4%), accuracy 98,3%. Đây là điểm
  vận hành thiên về an toàn.
- Hạ ngưỡng về 0,1615 (EER val) đổi 8 false accept lấy giảm FRR còn 2,0%;
  accuracy 98,2% — gần như cùng mức, nên **0,30 vẫn hợp lý** cho eKYC.
- EER 2,0% và AUC 0,9918 cho thấy embedding ArcFace tách genuine/impostor tốt;
  khoảng cách trung vị ~0,58 là rất lớn.
- **Không suy rộng sang CCCD:** LFW là ảnh đã căn, frontal, không có glare/
  perspective/print; selfie-CCCD thật sẽ khó hơn. Đây là đánh giá **recognition
  trên cặp chuẩn**, không phải end-to-end selfie-vs-CCCD.
- **Bug đã sửa:** fallback `_aligned_embedding` từng lặp qua crop của
  `detect_faces()` thay vì box của `detect_face_boxes()`, gây `TypeError` khi
  MediaPipe không có landmark. Đã sửa trong `face_service.py` + regression test
  `test_aligned_embedding_falls_back_to_detected_box`.

---

## 8. Checklist để bài báo "hoàn chỉnh" — phần còn thiếu

| # | Thí nghiệm | Dữ liệu cần | Trạng thái | Chỉ số sẽ báo cáo |
|---|---|---|---|---|
| 1 | FAS in-dataset | **CASIA-FASD crops** (`ai/data/archive (1)`, 73 MB; 2.660 frame/360 video sau dedup) | ✅ **Đã chạy — §6** | APCER/BPCER/EER frame + video |
| 1b | FAS robustness (JPEG/blur/downscale/dark) | cùng 2.660 crop, không cần tải thêm | ✅ **Đã chạy — §6.5** | APCER/BPCER/EER theo từng điều kiện |
| 2 | FAS in ấn (tùy chọn) | **NUAA** (390 MB) hoặc CASIA-FASD video gốc (~2,0 GB) | Không bắt buộc | APCER print (và theo loại) |
| 3 | Khớp khuôn mặt | **LFW deep-funneled** (`ai/data/lfw-deepfunneled`, View 1) | ✅ **Đã chạy — §7** | FAR/FRR/EER + TAR@FAR |
| 4 | OCR chính xác | 50–100 ảnh CCCD thật + transcription (đồng thuận) | Cần thu thập | CER/WER theo trường |
| 5 | QR thực tế | 20–30 ảnh mặt sau CCCD + payload kỳ vọng | Unit test đã có | tỉ lệ decode đúng |
| 6 | Quyết định đầu-cuối | 30–50 session selfie + CCCD + QR | **Chặn bởi (a)** | accuracy/ROC của policy |
| 7 | Ablation cờ pipeline | cùng bộ (6) | Chờ (6) | Δ accuracy khi tắt QR/FAS/tamper |

**(a) Bắt buộc trước thí nghiệm (6):** `verify_service` hiện auto-REJECT khi
tamper vượt ngưỡng, trong khi FPR test = **0.5967** → gần 60% người thật bị
từ chối. Phải đổi tamper sang **advisory** (hoặc nâng ngưỡng theo FPR 5–10%)
rồi mới đo E2E, nếu không kết quả sẽ vô nghĩa.

**Lệnh tải dữ liệu (người dùng tự chạy — trợ lý không tải mạng):**

> Bản crops 73 MB đã tải và đã chạy (§6); LFW deep-funneled đã có ở
> `ai/data/lfw-deepfunneled` (§7). Các lệnh dưới chỉ cần nếu muốn bản video gốc
> (tách print/replay + có bước detect) hoặc bộ in ấn bổ sung.
>
> **Bản chính thống 600 video (CBSR/ CASIA):** tải [Agreement PDF](http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf),
> ký, upload tại `http://www.cbsr.ia.ac.cn/images/xyz/AgreementUpload.aspx`
> (nếu lỗi, email bản scan tới `hjzhang@cbsr.ia.ac.cn`) → nhận account và tải
> trong 48 giờ. Trang CSDL: <http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp>.

```powershell
New-Item -ItemType Directory -Force ai\data\raw | Out-Null

# CASIA-FASD (~2,0 GB) — cần tài khoản Kaggle (Settings → Create New Token),
# đặt kaggle.json vào %USERPROFILE%\.kaggle\
#   https://www.kaggle.com/datasets/immada/casia-fasd
pip install kaggle
kaggle datasets download -d immada/casia-fasd -p ai\data\raw --unzip

# NUAA (390 MB, không cần tài khoản) — mở trang chủ, chọn format (1):
#   https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html

# LFW deep-funneled (đã có ở ai/data/lfw-deepfunneled) — chỉ tải lại nếu cần:
#   http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
# Các file matchpairsDevTrain/Test.csv + mismatchpairsDevTrain/Test.csv đã có
# trong ai/data (dùng trực tiếp, không cần pairs.txt).
```

> **Replay-Attack** (Zenodo 4593128): record bị **restricted** — phải đăng nhập và
> gửi yêu cầu EULA với email học thuật, người ký phải giữ chức vụ thường trực
> (sinh viên cần giảng viên đứng tên). Dung lượng chỉ vài GB (1.300 clip
> 320×240), không phải 3 TB; nếu bạn thấy "3 TB" là nhầm record khác. Chỉ dùng
> khi giảng viên hướng dẫn ký được EULA; trước mắt CASIA-FASD thay thế tốt.

Lưu ý license: CASIA-FASD / NUAA / Replay-Attack đều chỉ dùng cho nghiên cứu
phi thương mại. Bản Kaggle là mirror không chính thức — trong luận văn trích bài
gốc (Zhang et al., 2012 — CASIA-FASD; Tan et al., 2010 — NUAA) và ghi rõ nguồn
tải. Gói crops ở §6 là bản cắt lại của CASIA-FASD, dùng split không chính thức
đã được dedup — ghi rõ khi trích dẫn. Không commit dữ liệu (`ai/data/` đã gitignore).

---

## 9. Chỉ mục artifact (dùng khi viết bài)

| File | Nội dung | Sinh bởi |
|---|---|---|
| `reports/tamper_test.json` | Bảng 3.2 (test, per-threshold, per-attack, IoU, latency) | `evaluation/run_tamper_eval.py` |
| `reports/tamper_test_scores.csv` | Điểm từng ảnh test (1.385 dòng) | nt |
| `reports/tamper_val_scores.csv` | Điểm từng ảnh val (284 dòng) | `training/calibrate_tamper.py --scores-csv` |
| `reports/tamper_ablation.json` | Bảng 3.4 (quét w) | `evaluation/ablate_tamper_weights.py` |
| `reports/latency.json` | Bảng 4 (độ trễ + độ phủ) | `evaluation/bench_pipeline.py` |
| `reports/fas_casia_crops.json` | §6.2/6.4 frame-level + latency (TTA bật) | `evaluation/run_fas_eval.py` |
| `reports/fas_casia_crops_video.json` | §6.2–6.3 video-level + theo quality | `evaluation/analyze_fas_scores.py` |
| `reports/fas_casia_crops_notta*.json` | §6.4 ablation TTA | nt + `AI_FAS_TTA=false` |
| `reports/fas_casia_crops_scores.csv` | Điểm 2.660 frame (tái phân tích) | `evaluation/run_fas_eval.py` |
| `reports/fas_robustness.json` | §6.5 frame/video × 9 điều kiện | `evaluation/eval_fas_robustness.py` |
| `reports/fas_robustness_scores.csv` | Điểm 23.940 dòng (9 × 2.660) | nt |
| `reports/face_lfw.json` | §7.2 (2.200 val / 1.000 test, FAR/FRR/EER/TAR@FAR) | `evaluation/run_face_eval.py` |
| `reports/face_lfw_scores.csv` | Điểm 3.200 cặp LFW (no-face + thời gian) | nt |
| `data/fas_casia/manifest.csv` | Nhãn + subject/quality/video/frame sau dedup | `evaluation/prepare_casia_crops.py` |
| `reports/tamper-fantasyid-findings.vi.md` | Phân tích sâu: shortcut, OOD, §7 fix tên file | viết tay |
| `models/tamper_calibration.json` | Ngưỡng chính thức từ val | `training/calibrate_tamper.py` |
| `models/fantasyid_val.txt` | Manifest val đóng băng (284 ảnh) | `training/migrate_fantasyid_val_manifest.py` |
| `reports/fas_smoke*.json` | Smoke FAS cũ (ảnh + video) — chỉ tham chiếu code | `evaluation/run_fas_eval.py` |

Ánh xạ chương: 4.1 dataset/split → §2 + `fantasyid_val.txt`; 4.2 tamper →
§3.1–3.3; 4.3 ablation → §3.4; 4.4 độ trễ → §4; 4.5 kiểm thử → §5; 4.6 FAS →
§6 (frame + video + ablation TTA + robustness); 4.7 khớp mặt LFW → §7.

---

## 10. Threats to validity (viết nguyên văn được)

1. **Một bộ dữ liệu, không phải CCCD Việt Nam.** FantasyID là thẻ tổng hợp đa
   ngữ (Anh/Nga/Kazakh/Trung), tấn công `face_text`/`digital`; kết luận không
   suy rộng trực tiếp cho CCCD thật.
2. **Domain shift & shortcut thiết bị** làm AUC < 0.5; đây là hiện tượng thật
   của dữ liệu, không phải lỗi tính toán — đã chứng minh bằng ablation và bảng
   theo thiết bị.
3. **Mask head không phát vùng** → IoU = 0 do *không dự đoán*, không đo được
   chất lượng định vị; cần phân tích riêng (kiểm tra head/regression loss).
4. **Độ phủ detector thấp trên mặt in** (43,5% ảnh không thấy mặt ở độ phân
   giải gốc) → nhánh face/selfie cần tiling trước khi kết luận.
5. **Độ trễ đo trên laptop** (GPU rời + OCR CPU), không đại diện server; đo
   lại trên môi trường triển khai nếu báo cáo SLA.
6. **Chưa có thí nghiệm E2E** và chưa có người thật tham gia; mọi kết luận
   hiện tại là component-level.
7. **FAS đo trên crop CASIA-FASD cũ** (256×256, nhãn không tách print/replay,
   không có bước detect): kết quả cao (video EER 3,3%) có thể lạc quan so với
   ảnh in CCCD thật; BPCER nhóm HR chỉ dựa trên 30 video thật → CI rộng.
8. **Gói crops không có split hợp lệ**; đã dedup và coi là một tập đánh giá
   duy nhất, không phải held-out theo subject của bản gốc.
9. **Robustness FAS dùng suy giảm tổng hợp** trên crop 256×256 đã nén, không
   phải ảnh chụp điện thoại thật; nhóm HR chỉ có 30 video thật nên CI rộng.
   Kết quả định hướng chất lượng, không phải cam kết SLA.
10. **LFW không phải held-out hoàn toàn và không cùng miền CCCD.** Ảnh LFW đã
    căn mặt, frontal, độ phân giải tốt; checkpoint ArcFace tiền huấn luyện có
    thể đã thấy LFW (không kiểm chứng được) nên FAR/FRR ở §7 là **cận trên lạc
    quan**. Không suy rộng sang cặp selfie–CCCD có glare/perspective/print.

---

## 11. Việc tiếp theo (đúng thứ tự)

1. ✅ **FAS đã chạy** (§6) trên bản crops, gồm **robustness** (§6.5) — không
   cần tải thêm. Nếu cần tách **print/replay** và đo cả bước detect, tải
   CASIA-FASD đầy đủ (~2,0 GB) rồi chạy lại `run_fas_eval.py` **không** có
   `--no-detect`.
2. ✅ **Blur gate đã triển khai** (`AI_FAS_BLUR_GATE`, `LOW_QUALITY`/`RETRY`,
   test `tests/test_liveness_gate.py`); còn lại là hiệu chỉnh ngưỡng trên ảnh
   điện thoại thật.
3. ✅ **LFW đã chạy** (§7): `evaluation/run_face_eval.py`, 2.200 cặp val +
   1.000 cặp test; còn lại (nếu muốn) là kiểm tra chéo trên cặp selfie–CCCD thật.
4. **Đổi tamper sang advisory** (hoặc `AI_TAMPER_THRESHOLD=0.675` cho FPR ~39%
   vẫn quá cao → advisory là lựa chọn đúng) trước khi đo E2E.
5. **Thu 30–50 CCCD thật** (đồng thuận, ẩn danh) + 10–20 ảnh tấn công
   in/replay → CER, QR decode, E2E.
6. Viết Chương 4 từ các bảng §3–§7; Chương 5 (thảo luận) từ §10.

---

*Mọi số liệu trong tài liệu này tái lập được bằng các lệnh ở §3–§7 trên đúng
snapshot dataset `ai/data/fantasyid` + `ai/data/archive (1)` +
`ai/data/lfw-deepfunneled` hiện có; riêng §6.5 sinh từ
`reports/fas_robustness.json` bằng `evaluation/eval_fas_robustness.py`, §7 sinh
từ `reports/face_lfw.json` bằng `evaluation/run_face_eval.py`.*
