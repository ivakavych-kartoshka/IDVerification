# Kết Quả Thực Nghiệm — Module Tamper Trên FantasyID (Public Dataset)

Ngày chạy: 22/10/2026. Máy: Windows, RTX 4060 Laptop, Python 3.11.9, torch 2.13.0+cu126.
Mục đích: lấy số liệu thật cho Chương 4 (luận văn) và kiểm tra mô hình có dùng được
không. **Kết luận ngắn: hiện KHÔNG nên để module tamper tự động REJECT.**

> Tổng hợp số liệu cho bài báo (bảng tamper + ablation + độ trễ + checklist còn
> thiếu): xem `reports/PAPER_EVIDENCE.vi.md`.

---

## 1. Dataset

| | |
| --- | --- |
| Tên | **FantasyID** — Korshunov et al., IJCB 2025 (Idiap) |
| Nguồn | https://zenodo.org/records/17063366 (v1, 2.6 GB, open license) |
| Vị trí local | `ai\data\fantasyid\` (đã tải + `prepare_fantasyid.py`; manifest val dùng đường dẫn tương đối nên di chuyển máy không hỏng) |
| Train (đã chuẩn bị) | 1899 ảnh = 633 bona fide + 1266 forged, 1091 mask GT |
| Test (đã chuẩn bị) | 1385 ảnh = 300 bona fide + 1085 forged, 911 mask GT |
| Val | 284 ảnh carve từ train theo `models/fantasyid_val.txt` (168 forged / 116 bona) |

**Đặc thù quan trọng của FantasyID** (đọc từ `README.md` của dataset):

- Train-val **chỉ có 2 loại tấn công `digital_1`, `digital_2`, đều là `face_text`**
  (vừa tráo mặt vừa sửa chữ).
- Test cố tình là **out-of-distribution**: `digital_3` (chỉ sửa chữ, cùng template),
  `facedancer` (chỉ tráo mặt, template lạ), `textdiffuserft_bfei` (chữ + template lạ).
- Bona fide test dùng thiết bị **`iphone15`** — thiết bị **không xuất hiện** ở train
  (train chỉ có `iphone15pro`, `huawei`, `scan`).

---

## 2. Giao thức

1. Hiệu chuẩn ngưỡng trên **val** → `training/calibrate_tamper.py`.
2. **Áp nguyên ngưỡng đó** lên test (không chọn lại trên test) →
   `evaluation/run_tamper_eval.py`.
3. Phân tích theo device / attack_type → `evaluation/analyze_tamper_scores.py`,
   `evaluation/diag_dataset_composition.py`.

Ngưỡng thu được trên val: **0.57** (F1 cao nhất), operating point FPR=10% → **0.675**.
Ngưỡng đang deploy trong service: 0.55.

---

## 3. Kết quả

### 3.1 Val (cùng phân bố với train)

| Chỉ số | Giá trị |
| --- | --- |
| F1 @ 0.57 | **0.8113** (P=0.770, R=0.857) |
| F1 @ FPR=10% (thr 0.675) | 0.7852 (P=0.900, R=0.696) |
| Model-only F1 @ 0.5 | 0.8000 |

### 3.2 Test (held-out, OOD) — số liệu thật

| Ngưỡng | F1 | Precision | Recall | ACC | **FPR** | FNR |
| --- | --- | --- | --- | --- | --- | --- |
| 0.55 (deploy) | 0.4481 | 0.6710 | 0.3364 | 0.3509 | **0.5967** | 0.6636 |
| 0.57 (val-best) | 0.4301 | 0.6603 | 0.3189 | 0.3379 | 0.5933 | 0.6811 |
| 0.675 (val FPR10) | 0.2839 | 0.6278 | 0.1834 | 0.2751 | 0.3933 | 0.8166 |

- **AUC = 0.3417**, AP = 0.7429 (tập mất cân bằng 78% forged).
- **AUC model-only = 0.319** → **dưới 0.5, tức mô hình chấm ảnh giả THẤP hơn ảnh thật**
  trên test. Đây là dấu hiệu học lệch (shortcut), không phải lỗi nhãn.
- Độ trễ module: median **252 ms**, p95 **461 ms** (engine torch, chạy CPU).

### 3.3 Theo loại tấn công (ngưỡng 0.55)

| Loại | n | điểm trung bình | **Recall** | model_prob TB |
| --- | --- | --- | --- | --- |
| `face` (tráo mặt — đã thấy ở train) | 150 | 0.594 | **79.3%** | 0.803 |
| `text` (chỉ sửa chữ — chưa thấy ở train) | 935 | 0.248 | **26.3%** | 0.342 |

→ Mô hình phát hiện được face swap nhưng **gần như bất lực với text-only inpainting**,
loại tấn công chiếm 86% test set và cũng là loại giống gian lận CCCD thực tế
(sửa số hiệu, ngày sinh).

### 3.4 Theo thiết bị — nguyên nhân gốc

| Device | Nhãn | n | tamper_score TB | model_prob TB | % Dương tính @0.55 |
| --- | --- | --- | --- | --- | --- |
| huawei | **bonafide** | 100 | **0.588** | **0.821** | **80.0%** |
| huawei | forged | 361 | 0.405 | 0.553 | 45.7% |
| iphone15 | bonafide | 100 | 0.444 | 0.627 | 56.0% |
| iphone15 | forged | 100 | 0.482 | 0.655 | 62.0% |
| scan | bonafide | 100 | 0.374 | 0.527 | 43.0% |
| scan | forged | 362 | 0.225 | 0.308 | 24.9% |
| iphone15pro | forged | 262 | 0.173 | 0.242 | 18.3% |

**Đọc bảng** (số tính lại sau khi sửa tên file — xem §7): trong cả 3 device có
đủ hai lớp, ảnh *thật* được chấm điểm **cao hơn hoặc bằng** ảnh *giả* (huawei
0.821 vs 0.553; scan 0.527 vs 0.308; iphone15 gần hoà 0.627 vs 0.655). Khoảng
cách **giữa các device** còn lớn hơn khoảng cách giữa hai lớp: forged
`iphone15pro` 0.242 < bonafide `scan` 0.527 < bonafide `huawei` 0.821. Ở test,
lớp bonafide chỉ toàn `iphone15` — device **không có trong train** — trong khi
forged trải trên cả device quen lẫn lạ; mô hình chấm ảnh thật của "device lạ"
thấp hơn ảnh giả của "device quen" → AUC < 0.5, FPR ~60%.

### 3.5 Hai phát hiện phụ

- **Mask head "chết"**: 1385/1385 ảnh không sinh vùng nào (`n_regions=0`) →
  Localization IoU = 0.0 **không phải lỗi đo**, mà là mô hình không localise được.
  Cần kiểm tra lại trọng số loss phần mask / ngưỡng 0.5 / `min_frac` lúc suy luận.
- **Tầng heuristic gần như vô dụng trên test**: `signal_score` TB = 0.037;
  blockiness bona 0.064 vs forged 0.156 (trên val từng đo được |Δmedian| = 0.473
  → hoá ra phần lớn là tín hiệu phụ của subgroup, không phải dấu vết giả mạo).
- detector mặt tìm thấy `photo_box` ở 92.4% ảnh (tốt), nên `boundary_energy` /
  `noise_step` vẫn được tính nhưng không phân biệt lớp.

---

## 4. Lỗi công cụ đã tìm ra và sửa

**`training/calibrate_tamper.py` âm thầm chạy trên 0 ảnh.** Manifest
`models/fantasyid_val.txt` lưu đường dẫn Windows chuẩn (`D:\...\bona_fide\x.jpg`),
còn `CardTamperDataset` nối chuỗi từ `--data` dạng `D:/...` → ra đường dẫn **trộn**
(`D:/datasets/...\bona_fide\x.jpg`). So sánh chuỗi không khớp → 0 ảnh được chấm,
nhưng script vẫn in `best F1 = 0.0000` và **ghi file JSON vô nghĩa**.

Đã sửa: thêm `_norm_path()` (normcase + abspath hai bên), in số đường dẫn đã khớp,
và **dừng với exit code 2** khi manifest không khớp ảnh nào hoặc không chấm được
ảnh nào. Xem [calibrate_tamper.py](ai/training/calibrate_tamper.py).

---

## 5. Bài học cho luận văn (viết được ngay)

1. **Không được chỉ báo số trên val.** Val/test cùng phân bố → F1 0.81; test thật
   → F1 0.45, FPR 0.60. Đây là ví dụ textbook về shortcut learning, nên đưa vào
   phần "Threats to validity".
2. **Báo cáo theo nhóm (device, attack type)**, không chỉ số tổng hợp — chính bảng
   device đã giải thích được AUC < 0.5.
3. Kết luận có tính xây dựng: module **đang ở chế độ hỗ trợ điều tra**, chưa đủ
   điều kiện để tự động từ chối; đề xuất (a) huấn luyện cân theo thiết bị + augment
   nhiễu/mờ theo device, (b) thêm tấn công text-only vào tập train, (c) hiệu chuẩn
   ngưỡng **theo từng thiết bị**, (d) sửa mask head và đo lại IoU, (e) tách nhãn
   thiết bị ngay trong `prepare_fantasyid.py` — **đã sửa**, xem §7.

---

## 6. Lệnh tái lập (reproduce)

```powershell
cd ai
$env:PYTHONIOENCODING="utf-8"   # bắt buộc: script in tiếng Việt, console cp1252 sẽ vỡ

# 1) Hiệu chuẩn ngưỡng trên val
python training/calibrate_tamper.py `
  --data data/fantasyid/prepared/train `
  --weights models/tamper_net.pth `
  --val-manifest models/fantasyid_val.txt `
  --manifest-csv data/fantasyid/prepared/train/manifest.csv `
  --out models/tamper_calibration.json

# 2) Đánh giá trên test (giữ nguyên ngưỡng đã hiệu chuẩn)
python evaluation/run_tamper_eval.py `
  --data data/fantasyid/prepared/test `
  --weights models/tamper_net.pth `
  --thresholds 0.55,0.57,0.675 `
  --manifest-csv data/fantasyid/prepared/test/manifest.csv `
  --scores-csv reports/tamper_test_scores.csv `
  --out reports/tamper_test.json

# 3) Phân tích nguyên nhân
python evaluation/diag_dataset_composition.py --prepared data/fantasyid/prepared
python evaluation/analyze_tamper_scores.py --csv reports/tamper_test_scores.csv --thr 0.55

# 4) Regression
python -m pytest tests -q     # 37 passed
```

**Sản phẩm tạo ra**: `models/tamper_calibration.json`, `reports/tamper_test.json`,
`reports/tamper_test_scores.csv`, và 3 script trong `ai/evaluation/`.

---

## 7. Đã sửa rò rỉ subgroup qua tên file (22/10/2026)

**Vấn đề**: `_unique_stem` cũ chỉ gắn tiền tố `<device>__` khi trùng stem, nên
file đầu tiên của mỗi stem giữ tên trần. Nhóm "không có device" này trùng với
nhãn forged (test: 100% forged) → mọi phân tích nhóm theo device bị lệch giả tạo.

**Sửa**: `prepare_fantasyid.py::_unique_stem` luôn prefix device; regenerate toàn
bộ `prepared/` (train 633+1266, 1091 mask; test 300+1085, 911 mask — khớp từng
ảnh với bản cũ). `models/fantasyid_val.txt` được migrate bằng
`training/migrate_fantasyid_val_manifest.py` (replay thuật toán tên cũ → map sang
stem mới), giữ **đúng cùng 284 ảnh val** (116 bona / 168 forged).

**Bố cục sau khi sửa** (`evaluation/diag_dataset_composition.py`):

| Split | device | bona | forged | %forged |
| --- | --- | --- | --- | --- |
| train | scan | 211 | 422 | 66.7% |
| train | huawei | 211 | 422 | 66.7% |
| train | iphone15pro | 211 | 422 | 66.7% |
| test | scan | 100 | 362 | 78.4% |
| test | huawei | 100 | 361 | 78.3% |
| test | iphone15pro | 0 | 262 | 100% |
| test | iphone15 | 100 | 100 | 50.0% |

Không còn nhóm `(no-device)`; train cân tuyệt đối theo device. Lệch duy nhất còn
lại (`iphone15pro` 100% forged ở test) là **thuộc tính thật của dataset**: bona
fide test toàn bộ dùng `iphone15`, nên `iphone15pro` chỉ nằm trong ảnh tấn công.

**Bất biến số liệu**: model không đọc tên file, nên đổi tên + migrate val không
làm đổi kết quả — calibration tái lập y hệt (284 ảnh, F1 0.8113 @ 0.570); test
eval cũng trùng khớp từng chỉ số (AUC 0.3417, AP 0.7429, F1 0.4481 @ 0.55). Sửa
tên chỉ loại bỏ confound *phân tích nhóm*, không đổi bài toán học.

Bản `prepared/` cũ giữ ở `data/fantasyid/prepared_v1/` để đối chiếu; xóa khi
không cần.
