# TECHSTACK — Hệ thống xác thực CCCD (eKYC)

Tài liệu giải thích **toàn bộ công nghệ** được dùng trong dự án: dùng để làm gì, vì sao chọn,
chạy ở đâu, và hạn chế của từng lựa chọn.

> Repo này hiện chứa AI service (`ai/`) + tài liệu (`README.md`, `plan.md`, `railway.json`).
> Hai tầng `fe/` (Next.js) và `be/` (NestJS) được mô tả theo tài liệu chuẩn của dự án.

---

## Mục lục

1. [Tổng quan & sơ đồ kiến trúc](#1-tổng-quan--sơ-đồ-kiến-trúc)
2. [Bảng techstack tổng hợp](#2-bảng-techstack-tổng-hợp)
3. [Tầng Frontend — Next.js](#3-tầng-frontend--nextjs)
4. [Tầng Backend — NestJS + Prisma + PostgreSQL](#4-tầng-backend--nestjs--prisma--postgresql)
5. [Tầng AI — FastAPI](#5-tầng-ai--fastapi)
6. [Bộ model AI (weights)](#6-bộ-model-ai-weights)
7. [Luồng dữ liệu 1 lần xác thực](#7-luồng-dữ-liệu-1-lần-xác-thực)
8. [Cơ chế fallback](#8-cơ-chế-fallback)
9. [Deploy: Docker · Railway · Hugging Face](#9-deploy-docker--railway--hugging-face)
10. [Kiểm thử](#10-kiểm-thử)
11. [Hạn chế đã biết](#11-hạn-chế-đã-biết)

---

## 1. Tổng quan & sơ đồ kiến trúc

```text
┌───────────────────────────────────────┐
│  Frontend — Next.js 15 / React 19     │  fe/, :3001
│  App Router · getUserMedia (webcam)   │
└────────────────┬──────────────────────┘
                 │ REST + JSON/multipart, Bearer JWT
                 ▼
┌───────────────────────────────────────┐        ┌──────────────────────┐
│  Backend — NestJS 10 + Prisma         │───────▶│ PostgreSQL           │
│  Auth (JWT/bcrypt) · Upload · Proxy   │        │ qua Prisma ORM       │
│  Verification orchestration · Swagger │        └──────────────────────┘
└────────────────┬──────────────────────┘
                 │ HTTP (multipart) — Frontend KHÔNG gọi thẳng AI
                 ▼
┌───────────────────────────────────────────────────────────────────┐
│  AI — FastAPI (uvicorn) , :8001                                  │
│  RapidOCR/PaddleOCR · OpenCV · UltraFace · MediaPipe · ArcFace   │
│  MiniFASNetV2 (passive) · Active liveness · TamperNet            │
└───────────────────────────────────────────────────────────────────┘
```

**Nguyên tắc phân quyền trách nhiệm:**

| Tầng | Chịu trách nhiệm | Không chịu trách nhiệm |
| --- | --- | --- |
| Next.js | UI, webcam, hiển thị kết quả | Gọi AI, lưu DB |
| NestJS | Auth, ownership, validate & lưu file, gọi AI, lưu kết quả, quyết định trạng thái | Suy luận CV/AI |
| FastAPI AI | OCR, QR, face, liveness, tamper, **ra quyết định AI** | Xác thực người dùng, lưu DB |

Lý do tách AI ra service riêng: model nặng (torch, mediapipe, onnxruntime) cần RAM/CPU khác
backend CRUD; có thể scale/deploy độc lập (Docker Space/Railway); và thay engine AI mà không
đụng code business.

---

## 2. Bảng techstack tổng hợp

### Ngôn ngữ & runtime

| Công nghệ | Phiên bản | Tầng | Vai trò |
| --- | --- | --- | --- |
| TypeScript / Node.js | Node ≥ 18 (khuyến nghị 20) | fe, be | Runtime backend & frontend |
| Python | ≥ 3.10 (Docker dùng 3.11) | ai | Runtime AI service |
| React | 19 | fe | UI library |
| Next.js | 15 (App Router) | fe | Framework web, SSR/routing |
| NestJS | 10 | be | Framework backend (DI, module, guard) |
| FastAPI | 0.128.0 | ai | Web framework AI, tự sinh OpenAPI |

### Dữ liệu & giao tiếp

| Công nghệ | Tầng | Vai trò |
| --- | --- | --- |
| PostgreSQL | be | DB chính (CCCD là dữ liệu nhạy cảm → self-host/local) |
| Prisma | be | ORM + migration + type-safe client (`schema.prisma`) |
| REST + multipart | fe↔be↔ai | Truyền JSON + file (front/back/selfie/liveness[]) |
| JWT (Bearer) | fe, be | Xác thực, ký bằng `JWT_SECRET`, hạn `JWT_EXPIRES_IN` |
| bcrypt (10 rounds) | be | Hash mật khẩu |
| Swagger / OpenAPI | be (`/api/docs`), ai (`/docs`) | Tài liệu API tự sinh |

### Computer Vision & AI (tầng `ai/`)

| Công nghệ | Phiên bản | Vai trò |
| --- | --- | --- |
| OpenCV (`opencv-python`) | 4.10.0.84 | Ảnh nền tảng: EXIF-rotate, warp align, QR, Laplacian |
| NumPy | 2.3.5 | Ma trận ảnh, tính toán |
| Pillow | 12.3.0 | Load ảnh, EXIF orientation |
| SciPy | 1.16.3 | Hỗ trợ xử lý số (OCR/post-process) |
| ONNX Runtime | 1.24.3 | Suy luận CPU các model ONNX (UltraFace, ArcFace, RapidOCR, TamperNet.onnx) |
| RapidOCR | 1.4.4 | **Engine OCR chính** (ONNX, ổn định) |
| PaddleOCR | 3.7.0 | OCR fallback, `lang="vi"` (backup - không dùng) |
| MediaPipe Tasks | 1.0.1 | FaceLandmarker: 5 landmark + blendshapes + head pose |
| PyTorch | 2.13.0 (+cu126 local / CPU trong Docker) | Chạy MiniFASNetV2 & TamperNet (.pth) |
| torchvision | 0.28.0+cu126 | Phụ trợ torch (train/transform) |
| albumentations ≥ 2.0.8 | train | Augmentation khi train TamperNet |
| scikit-image | 0.25.2 | Tiền/xử lý ảnh khi train & calibrate |
| pyzbar (optional) | — | Fallback decode QR nếu cài |

### Deploy & kiểm thử

| Công nghệ | Vai trò |
| --- | --- |
| Docker (`python:3.11-slim-bookworm`) | Đóng gói AI service |
| Hugging Face Docker Space | Host AI (CPU, 16GB RAM, port 7860) |
| Railway (`railway.json`) | Deploy AI bằng Dockerfile, healthcheck `/healthz` |
| uvicorn | ASGI server, `--workers 1` (tránh nhân bản RAM model) |
| pytest | Test AI (`ai/tests/`) |
| Jest | Test backend (unit/e2e) |

---

## 3. Tầng Frontend — Next.js

**Vị trí:** `fe/` · cổng `3001`

| Thành phần | Chi tiết |
| --- | --- |
| Next.js 15 App Router | Routing theo thư mục `app/` (`/`, `/login`, `/register`, `/kyc`, `/verifications`) |
| React 19 | Component function + hooks |
| CSS thuần | `app/globals.css`, giao diện tối — không kéo UI library → nhẹ, ít dependency |
| `lib/api.ts` | Client REST duy nhất, tự gắn `Authorization: Bearer <token>` |
| localStorage | Lưu JWT (`cccd_token`) |

### Webcam / liveness phía client — `components/LivenessChallenge.tsx`

| API trình duyệt | Dùng để |
| --- | --- |
| `navigator.mediaDevices.getUserMedia` | Mở webcam (cần quyền + HTTPS/localhost) |
| `<canvas>` → `toDataURL` | Chụp frame **320×240 JPEG** mỗi **550 ms** |
| `POST /api/ai/liveness/pose` (qua NestJS) | Gửi từng frame để lấy yaw → hướng dẫn realtime |

- Chuỗi thử thách trên UI: **nhìn thẳng → quay trái → quay phải** (ngưỡng giống AI:
  `STRAIGHT_MAX = 12`, `TURN_MIN = 18`).
- Frame "nhìn thẳng" đầu tiên được lưu làm **selfie** để so với ảnh trên thẻ.
- Ảnh lịch sử được tải qua endpoint có auth rồi tạo `ObjectURL` (không phát URL công khai).

**Vì sao làm hướng dẫn realtime bằng 1 endpoint pose riêng?** Để người dùng biết mình đã
quay đủ chưa ngay trên UI, thay vì submit xong mới bị reject → giảm tỉ lệ thất bại.

---

## 4. Tầng Backend — NestJS + Prisma + PostgreSQL

**Vị trí:** `be/` · cổng `3000` · base `/api` · Swagger `/api/docs`

| Module | Đường dẫn | Việc làm |
| --- | --- | --- |
| `auth/` | `/api/auth/*` | register / login / me — JWT + bcrypt |
| `ai/` | `/api/ai/*` | **Proxy** sang AI: `/health`, `/liveness/pose` |
| `storage/` | — | Ghi file vào `UPLOAD_DIR/<userId>/<uuid>.<ext>` |
| `verification/` | `/api/verifications/*` | Nghiệp vụ eKYC chính |
| `prisma/` | — | `PrismaService` (kết nối, pool) |

### Vì sao Frontend không gọi AI trực tiếp?

1. **Kiểm soát auth/ownership** ở một nơi duy nhất.
2. AI không biết gì về user → không thể ghi DB hay kiểm quyền.
3. Không lộ URL/service AI ra trình duyệt (CORS `*` của AI chỉ dành cho backend).
4. Có chỗ tập trung validate (MIME, ≤ 8 MB, ≤ 40 frame), chống trùng số CCCD.

### Chi tiết kỹ thuật

| Hạng mục | Công nghệ / quy tắc |
| --- | --- |
| Upload | `FileFieldsInterceptor` (multer của NestJS): `front` (bắt buộc), `back?`, `selfie?`, `liveness[]` |
| Giới hạn | 8 MB/file, tối đa 40 frame liveness; MIME chỉ nhận `jpeg/jpg/png/webp` |
| Lưu file | Tên UUID ngẫu nhiên, ngoài web root, phân vùng theo `userId` |
| Gọi AI | HTTP client + `AI_SERVICE_TIMEOUT_MS=60000`; timeout → `504`, không reachable → `503` |
| Mock | `AI_MOCK=true` → trả `VERIFIED` giả lập để dev không cần bật AI |
| Trạng thái | record đi `PENDING → PROCESSING → VERIFIED / REJECTED / ERROR` |
| Chống trùng | `documentNumber` đã có record `VERIFIED` khác → từ chối |
| Ownership | Mọi `findOne`/`preview`/`retry` đều so `userId`, khác user → `403` |
| An toàn | AI lỗi → **không bao giờ** trả `VERIFIED` (chỉ `ERROR`) |

### Prisma schema (PostgreSQL)

| Model | Ý nghĩa |
| --- | --- |
| `User` | `id(cuid)`, `email @unique`, `password`(bcrypt), `fullName`, `role`, `verifications[]` |
| `IdentityVerification` | `userId` (FK cascade), `documentType`, `documentNumber`(index), đường dẫn 3 ảnh, `status`(index), các field JSON `ocrResult/documentQuality/faceMatchResult/livenessResult/aiRawResult`, `rejectionReason`, relation `qr` 1-1 |
| `VerificationQr` | 1 dòng/verification — lưu payload & field QR đã parse + `consistencyStatus/matchedFields/mismatchedFields` để **audit độc lập** với JSON AI |
| Enum `VerificationStatus` | `PENDING · PROCESSING · VERIFIED · REJECTED · ERROR` |

Lệnh: `prisma:generate` · `prisma:migrate` · `prisma:push` · `prisma:studio` · `seed`.

**Vì sao QR lưu riêng 1 bảng?** JSON của AI trả về mỗi lần là blob — tách ra bảng quan hệ
cho phép truy vấn/audit (ví dụ tìm mọi verification có QR mismatch) mà không phải parse JSON.

---

## 5. Tầng AI — FastAPI

**Vị trí:** `ai/` · cổng `8001` · Swagger `/docs` · health `/healthz`

```text
ai/app/
├── main.py           # FastAPI + CORS(*) + mount 6 router
├── routers/          # health, ocr, qr, verify, liveness, tamper
└── services/         # ocr, qr, face, liveness, fasnet,
                      # active_liveness, tamper, tamper_net, verify
```

### 5.0 Web layer

| Công nghệ | Vì sao |
| --- | --- |
| FastAPI | Async routing + tự sinh OpenAPI; dependency injection đơn giản |
| `uvicorn[standard]` | ASGI server tốc độ cao (reload khi dev) |
| `python-multipart` | Nhận `multipart/form-data` (file upload) |
| `run_in_threadpool` | **Suy luận nặng chạy trong threadpool** → event loop không bị block, endpoint vẫn phản hồi |
| CORS `allow_origins=["*"]` | AI là service nội bộ, do backend gọi (không để trình duyệt gọi thẳng) |

### 5.1 OCR — `services/ocr_service.py`

| Hạng mục | Chi tiết |
| --- | --- |
| Engine chính | **RapidOCR** (`rapidocr-onnxruntime`) — ONNX Runtime, ổn định trên mọi OS |
| Fallback | **PaddleOCR 3.7** `lang="vi"` (nặng hơn, dễ lệch môi trường — bỏ trong Docker) |
| Tiền xử lý | **EXIF-rotate** khi load (Pillow) → ảnh chụp thẳng từ điện thoại không bị quay sai |
| Đầu ra | `lines[{text, score}]`, `full_text`, `document_number` |
| Trích số CCCD | `_find_document_number`: ưu tiên **12 chữ số**, rồi CMND cũ **9 chữ số**, tránh nhầm **8 chữ số ngày tháng** |

**Vì sao ưu tiên RapidOCR?** Cùng kết quả OCR tiếng Việt nhưng nhẹ hơn PaddleOCR ~1GB
(PaddlePaddle framework), không cần thư viện hệ thống đặc thù, chạy ONNX Runtime chung
runtime với UltraFace/ArcFace → một dependency duy nhất cho mọi suy luận.

### 5.2 QR cross-check — `services/qr_service.py`

| Hạng mục | Chi tiết |
| --- | --- |
| Decoder chính | OpenCV `QRCodeDetector` + `QRCodeDetectorAruco` |
| Fallback | `pyzbar` (nếu cài) |
| Tăng tỉ lệ đọc | Nhiều biến thể tiền xử lý: resize, grayscale, upscale, adaptive threshold |
| Payload CCCD | `<số CCCD>\|<số cũ?>\|<họ tên>\|<ngày sinh>\|<giới tính>\|<quê quán>\|<ngày cấp>` |
| Parser | Chịu được thiếu field — phân loại field **theo hình dạng**, không theo vị trí cố định |
| So khớp | `compare()` với text OCR → `NOT_FOUND / MATCH / PARTIAL / MISMATCH` |

Cấu hình: `AI_QR_ENABLED=true` · `AI_QR_REQUIRE=false` · `AI_QR_STRICT=true` (số CCCD trên QR
phải xuất hiện trong OCR) · `AI_QR_STRICT_NAME=false`.

> **Lưu ý bảo mật:** QR in trên thẻ **không phải bằng chứng mật mã** (có thể photocopy) — nó
> chỉ là **tín hiệu chống giả mạo** chéo (inconsistency = dấu hiệu sửa thẻ).

### 5.3 Face — `services/face_service.py`

Pipeline 3 model, mỗi model một việc:

| Bước | Model/Công nghệ | Việc làm |
| --- | --- | --- |
| 1. Detect | **UltraFace** `version-RFB-640.onnx` | Tìm face box. Input cố định **480×640 (H×W)**, BGR `(x-127.5)/127.5`, conf ≥ 0.6, NMS IoU 0.4 |
| 1'. Fallback detect | **Haar Cascade** `haarcascade_frontalface_default.xml` | Khi onnxruntime lỗi |
| 2. Landmark | **MediaPipe FaceLandmarker** `face_landmarker.task` | 5 điểm chuẩn (2 mắt, mũi, 2 khóe miệng) |
| 3. Align | `cv2.estimateAffinePartial2D` (method LMEDS) | Warp về template **ArcFace 112×112** chuẩn |
| 4. Embed | **ArcFace** `recognition.onnx` | Vector 112×112, `(x-127.5)/127.5`, **L2-normalize** |
| 5. Match | Cosine similarity | `≥0.30` (`AI_MATCH_SCORE`) → MATCH · `≥0.20` → PARTIAL · dưới → NO_MATCH |

**Chi tiết quan trọng:**

- **Tiling** (`_tile_images`, tile 800px, overlap 50%): ảnh thẻ CCCD chụp độ phân giải cao →
  ảnh chân dung in chỉ vài chục pixel → MediaPipe/UltraFace 1 pass trên toàn ảnh sẽ bỏ sót.
  Thứ tự: thử landmark trên từng tile → nếu không có, thử detect box trên từng tile.
- **Selfie phải đúng 1 mặt**: `detect_face_boxes(selfie)` > 1 → `multiple_faces` → REJECTED.
- Lệch hai kênh màu: input **RGB** everywhere, chỉ đổi sang BGR đúng lúc pre-process UltraFace.

### 5.4 Passive liveness — `services/liveness_service.py` + `fasnet.py`

| Hạng mục | Chi tiết |
| --- | --- |
| Model | **MiniFASNetV2** `2.7_80x80_MiniFASNetV2.pth` (PyTorch) |
| Output | 3 class: index **1 = thật**, **0/2 = giả** (print/replay) |
| Input | Crop 80×80 quanh face box, mở rộng hệ số **×2.7** (`AI_FAS_SCALE`) |
| Tiền xử lý ⚠️ | BGR trong **[0, 255]** — **không** chia `/255`. Chia về [0,1] → mọi mặt thật bị coi là giả |
| Ngưỡng | `AI_FAS_REAL_THRESHOLD=0.5` (xác suất class "thật" tối thiểu) |
| TTA | `AI_FAS_TTA=true` — average verdict với ảnh lật ngang → kết quả ổn định |
| Device | `AI_FAS_DEVICE=auto` → CUDA nếu có, ngược lại CPU |
| Fallback | Thiếu torch/checkpoint hoặc lỗi suy luận → **Laplacian variance** blur, ngưỡng `25.0` |

**Nguyên lý:** đây là kiểm tra **texture** — ảnh in/màn hình có phổ tần/hoa văn khác mặt
thật (moiré, độ mịn giấy in, phản xạ màn hình).

### 5.5 Active liveness — `services/active_liveness_service.py`

| Hạng mục | Chi tiết |
| --- | --- |
| Nguồn pose | MediaPipe FaceLandmarker → **yaw** từng frame (+ blendshapes cho blink) |
| Thử thách | Nhìn thẳng → quay trái → quay phải |
| Ngưỡng | `\|yaw\| ≤ 12°` = thẳng · `\|yaw\| ≥ 18°` = đã quay · mặt phải có ở `≥80%` frame · biên độ yaw `≥18°` |
| **Chống ảnh tĩnh** | `_is_frozen()`: gray 64×64 → `mean(\|Δ\|)` giữa frame liên tiếp → `max < AI_ACTIVE_STATIC_DIFF (1.5)` → `"Frames appear static (possible photo/screen replay)"` |
| Blink | Từ blendshapes: `blink ≥ 0.5` = nhắm, `≤ 0.2` = mở; `AI_ACTIVE_REQUIRE_BLINK=true` để bắt buộc |

**Chống ảnh tĩnh 2 tầng:** (a) xét frame gần như không đổi; (b) bản thân logic — ảnh tĩnh
không thể tạo ra **hai hướng yaw trái/phải**.

### 5.6 Tamper detection — `services/tamper_service.py` + `tamper_net.py`

Phát hiện **chỉnh sửa số** trên ảnh thẻ (đổi chân dung, inpaint trường chữ) — khác QR
(độ nhất quán **dữ liệu**) và liveness (bằng chứng **vật lý**).

**Mô hình chính: TamperNet** (~492K tham số, cảm hứng TruFor CVPR'23 & DeepID@ICCV 2025):

| Stream | Công nghệ | Bắt tín hiệu gì |
| --- | --- | --- |
| RGB | Conv thông thường | Bố cục / ngoại hình |
| Forensic | **SRM 5×5 high-pass (4 filter, cố định — không học)** | Dấu resampling, blending |
| Frequency | **Block-DCT 8×8** AC/DC energy ratio (khác biệt hóa) | Sai lệch nén vùng ghép |
| Heads | `logit` (classification) + `mask_logits` (localization per-pixel → bounding box) | Bona fide vs forged + vùng nghi ngờ |

**Fallback cổ điển (4 tín hiệu cv2/ numpy):** `boundary_energy` (đường ghép), `noise_step`
(discontinuity sàn nhiễu), `blockiness` (lưới JPEG lệch), `low_hf_fraction` (mất cao tần do inpaint).

**Fusion:** `score = w × model + (1-w) × heuristic`, `AI_TAMPER_MODEL_WEIGHT=0.70`, chuẩn hóa
lại khi thiếu tín hiệu (mặt sau thẻ không có chân dung).

**Quy tắc an toàn (quan trọng):** engine **chỉ heuristic không bao giờ** trả `TAMPERED`,
chỉ tối đa `SUSPICIOUS` — reject cứng phải có model neural đã load thật. Mọi kết quả trả
`model_loaded` để fallback im lặng không bao giờ làm yếu quyết định bảo mật.

| Env | Mặc định |
| --- | --- |
| `AI_TAMPER_ENABLED` | `true` |
| `AI_TAMPER_ADVISORY` | `true` (kết quả `TAMPERED` chỉ để tham khảo, không chặn; đặt `false` để chặn lại) |
| `AI_TAMPER_MODEL` | `models/tamper_net.pth` (đổi sang `.onnx` → chuyển backend onnxruntime) |
| `AI_TAMPER_THRESHOLD` | `0.55` |
| `AI_TAMPER_SUSPECT` | `0.35` |
| `AI_TAMPER_MODEL_WEIGHT` | `0.70` |

**Training:** dataset **FantasyID** (DeepID@ICCV 2025, Zenodo CC BY 4.0) — 786 bona fide +
1572 forged (train), 300 + 1085 (test); metric **F1 @ 0.5**. Scripts: `training/prepare_fantasyid.py`
(grayscale homography từ `cropping_info` trước khi rasterize mask), `train_tamper.py`,
`calibrate_tamper.py` (sweep ngưỡng trên test split), `export_tamper_onnx.py` (opset 18).

### 5.7 Quyết định cuối cùng — `services/verify_service.py`

```text
VERIFIED  ⇔ face_match == MATCH
         AND passive_liveness.is_live == true
         AND active_liveness.status == PASS
         AND không có QR reject
         AND selfie đúng 1 khuôn mặt
         AND tamper không phải TAMPERED

REJECTED  ⇐ bất kỳ điều kiện nào sai (kèm rejection_reason)
PROCESSING = trạng thái trung gian
```

Đầu ra: `status`, `ocr`, `qr`, `qr_consistency`, `document_quality`, `face`, `face_match`,
`selfie_face_count`, `multiple_faces`, `passive_liveness`, `active_liveness`, `tamper`,
`rejection_reason`.

### 5.8 Endpoints AI

| Method | Path | Mô tả |
| --- | --- | --- |
| GET | `/healthz` | Health (dùng làm healthcheck Docker/Railway) |
| GET | `/` | Root check |
| POST | `/ocr/extract` | OCR 1 ảnh → lines + full_text + document_number |
| POST | `/qr/extract` | Giải mã QR → payload + fields |
| POST | `/verify/` | Multipart `front, back?, selfie?, liveness[]` → kết quả đầy đủ |
| GET | `/liveness/healthz` | Health liveness |
| POST | `/liveness/pose` | Head pose 1 frame (dùng cho UI realtime) |
| POST | `/liveness/active` | Kiểm tra chuỗi thách thức |
| POST | `/tamper/analyze` | Phân tích tamper 1 ảnh (độc lập `/verify/`) |
| GET | `/tamper/healthz` | Engine hiện tại + model đã load chưa |

---

## 6. Bộ model AI (weights)

| File | Dung lượng | Framework/Format | Vai trò |
| --- | --- | --- | --- |
| `version-RFB-640.onnx` | ~1.5 MB | ONNX | **UltraFace** — detect khuôn mặt (input 480×640) |
| `face_landmarker.task` | ~3.6 MB | MediaPipe Tasks | **5 landmark + blendshapes (blink) + head pose (yaw/pitch/roll)** |
| `recognition.onnx` | ~13 MB | ONNX | **ArcFace** — embedding 112×112 để so khớp |
| `2.7_80x80_MiniFASNetV2.pth` | ~1.8 MB | PyTorch | **Passive anti-spoofing** (print/replay) |
| `tamper_net.pth` | ~2 MB | PyTorch | **TamperNet** — phát hiện chỉnh sửa ảnh thẻ |

Tổng ~22 MB, **track trong git** (không cần tải riêng) → image build & deploy đơn giản.

**Tại sao ONNX cho 3 model?** Suy luận CPU nhanh qua `onnxruntime`, không cần framework
lớn, cùng runtime với OCR → chỉ 1 engine inference cho phần lớn pipeline. PyTorch chỉ giữ
cho 2 model `.pth` (MiniFASNetV2, TamperNet) vì checkpoint gốc là `.pth`.

---

## 7. Luồng dữ liệu 1 lần xác thực

```text
[FE] getUserMedia → frame 320×240 @550ms
      └─ POST /api/ai/liveness/pose ──▶ [BE proxy] ──▶ [AI] MediaPipe yaw
                                                        → UI hướng dẫn pha tiếp theo
[FE] upload front(+back) + selfie + liveness[]
      └─ POST /api/verifications (multipart, Bearer JWT)
           [BE] validate MIME/8MB/≤40 frame → ghi file UUID theo userId
                → record PENDING → PROCESSING
                → POST AI /verify/
                     [AI] ① OCR (RapidOCR→Paddle)  (front + back)
                          ② QR decode (OpenCV→pyzbar) + compare với OCR
                          ③ document quality (Laplacian blur)
                          ④ face: UltraFace→MediaPipe→align→ArcFace
                                  selfie vs ảnh trên thẻ → cosine
                          ⑤ passive liveness: MiniFASNetV2 (fallback Laplacian)
                          ⑥ active liveness: yaw sequence + frozen-frame + blink
                          ⑦ tamper: TamperNet + 4 heuristic signals
                          ⑧ decision → VERIFIED / REJECTED
                [BE] map status, chống trùng documentNumber,
                     lưu JSON + bảng VerificationQr → PENDING done
      ◀── { status, rejectionReason, ... }
[FE] hiển thị VERIFIED / REJECTED + lịch sử
```

---

## 8. Cơ chế fallback

Mọi engine đều có đường dự phòng để service **không chết** khi thiếu model/phụ thuộc:

| Thành phần | Chính | Fallback | Ghi chú |
| --- | --- | --- | --- |
| OCR | RapidOCR | PaddleOCR (`lang="vi"`) | Đảo thứ tự theo môi trường; Docker bỏ Paddle |
| Face detect | UltraFace ONNX | Haar Cascade | Khi onnxruntime lỗi |
| Face landmark | MediaPipe | Bỏ qua → dùng box crop + resize 112×112 | `_aligned_embedding` bước 2 |
| Passive liveness | MiniFASNetV2 | Laplacian variance ≥ 25.0 | Auto-skip nếu thiếu torch/checkpoint |
| QR | OpenCV QRCodeDetector | pyzbar | pyzbar optional |
| Tamper | TamperNet + heuristic | Heuristic-only (chỉ tới `SUSPICIOUS`) | **Không bao giờ** reject cứng 1 mình |
| Backend → AI | Gọi thật | `AI_MOCK=true` → `VERIFIED` | Chỉ cho dev/test |

Test tự **skip** phần liên quan khi thiếu torch/checkpoint → CI/CD không fail vì thiếu model.

---

## 9. Deploy: Docker · Railway · Hugging Face

### Docker (`ai/Dockerfile`)

| Hạng mục | Giá trị | Vì sao |
| --- | --- | --- |
| Base image | `python:3.11-slim-bookworm` | Nhẹ, ổn định |
| Hệ thống | `libgl1`, `libglib2.0-0`, `libgomp1`, `ffmpeg`, `libsm6`, `libxext6` | opencv (dù headless), onnxruntime/torch (OpenMP), mediapipe |
| Torch | Cài riêng từ `download.pytorch.org/whl/cpu` | Tránh kéo bản CUDA ~2.5GB (không có GPU) |
| Deps | `requirements-docker.txt` | `opencv-python-headless`, **không PaddleOCR** (tiết kiệm ~1GB), không dep train |
| User | `useradd -u 1000 user` | HF Space yêu cầu uid 1000 cho Dev Mode |
| Port | `PORT=7860` mặc định, env `PORT` override | HF đọc port từ YAML README; Railway inject `$PORT` |
| Healthcheck | `curl /healthz`, start-period 120s | Model nạp lần đầu chậm |
| Workers | `--workers 1` | Không nhân đôi RAM model |

> **2 file requirements:** `requirements.txt` (môi trường local/train, torch CUDA + PaddleOCR +
> albumentations) và `requirements-docker.txt` (runtime deploy, CPU-only).

### Railway — `railway.json`

```json
build:    Dockerfile builder, dockerfilePath = ai/Dockerfile, buildContext = ai
deploy:   healthcheckPath = /healthz, timeout 300s,
          restartPolicy ON_FAILURE, maxRetries 3
```

### Hugging Face Docker Space

Push thư mục `ai/` (Dockerfile + requirements-docker.txt + README YAML `sdk: docker`,
`app_port: 7860` + `app/` + `models/`) → host miễn phí CPU 2 vCPU/16GB.

### Yêu cầu môi trường chạy local

- Node ≥ 18 · Python ≥ 3.10 · PostgreSQL chạy sẵn · webcam hỗ trợ `getUserMedia`
- Thứ tự khởi động: **AI (8001) → BE (3000) → FE (3001)**

---

## 10. Kiểm thử

| Tầng | Framework | Lệnh | Phạm vi |
| --- | --- | --- | --- |
| AI | **pytest** | `cd ai && python -m pytest -v` | crop geometry, regression input [0,255] của MiniFASNet, parse QR + cross-check, parse số CCCD, post-process UltraFace, active liveness (no-frame, frozen), kiến trúc TamperNet + contract heuristic không tự reject |
| Backend | **Jest** | `cd be && npm test` (+ `test:e2e`) | Unit + e2e |
| Fixture QR | `qrcode[pil]` | `python tests/make_qr_fixture.py "<payload>" back_qr.png` | Tạo ảnh QR test |

Các module test: `test_liveness.py`, `test_active_liveness.py`, `test_qr_service.py`,
`test_ocr_service.py`, `test_face_service.py`, `test_tamper.py`.

---

## 11. Hạn chế đã biết

| Hạng mục | Hạn chế | Ở đâu ghi |
| --- | --- | --- |
| Passive liveness | Không phát hiện **mask 3D / mannequin**; không phát hiện **video replay** | `README.md`, `ai/README.md` |
| Active liveness | Bắt được ảnh/ảnh tĩnh & screen replay **đứng yên**; video replay có diff thật vẫn qua được → chỉ nhờ ràng buộc yaw trái/phải | `active_liveness_service.py` |
| QR | Photocopy được → **không phải bằng chứng mật mã** | `qr_service.py` / README |
| Tamper heuristic | Threshold (`0.55/0.35`) và `SIGNAL_NORM` là **placeholder** tới khi calibrate trên data thật; engine heuristic-only không được reject | `ai/README.md` |
| Tamper data | Split `val` của FantasyID cần license riêng → dùng split nội bộ từ `train` để early stop | `ai/README.md` |
| Face | Chỉ so khớp 2D — không có challenge chống **in ảnh + nhại pose** ngoài active liveness | — |
| CORS AI | `allow_origins=["*"]` — an toàn vì AI là service nội bộ, do backend gọi | `main.py` |

---

## Phụ lục — Biến môi trường (tóm tắt)

**AI service**

| Env | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_FAS_ENABLED` / `AI_FAS_MODEL` / `AI_FAS_SCALE` | `true` / `models/2.7_80x80_...pth` / `2.7` | MiniFASNetV2 |
| `AI_FAS_REAL_THRESHOLD` / `AI_FAS_DEVICE` / `AI_FAS_TTA` | `0.5` / `auto` / `true` | Ngưỡng / device / TTA |
| `AI_ACTIVE_STATIC_CHECK` / `AI_ACTIVE_STATIC_DIFF` | `true` / `1.5` | Chống frame tĩnh |
| `AI_ACTIVE_REQUIRE_BLINK` | `false` | Bắt buộc nháy mắt |
| `AI_QR_ENABLED` / `AI_QR_REQUIRE` / `AI_QR_STRICT` / `AI_QR_STRICT_NAME` | `true` / `false` / `true` / `false` | Chính sách QR |
| `AI_MATCH_SCORE` / `AI_PARTIAL_SCORE` | `0.30` / `0.20` | Ngưỡng cosine face |
| `AI_TAMPER_ENABLED` / `MODEL` / `THRESHOLD` / `SUSPECT` / `MODEL_WEIGHT` | `true` / `tamper_net.pth` / `0.55` / `0.35` / `0.70` | Tamper |

**Backend** — `DATABASE_URL` · `JWT_SECRET` · `JWT_EXPIRES_IN=1d` · `PORT=3000` ·
`AI_SERVICE_URL=http://127.0.0.1:8001` · `AI_SERVICE_TIMEOUT_MS=60000` · `UPLOAD_DIR=uploads` ·
`AI_MOCK=false`

**Frontend** — `NEXT_PUBLIC_API_URL=http://localhost:3000/api`

> `.env` không commit (đã có `.gitignore`). **Đổi `JWT_SECRET` mặc định** khi triển khai thật.

---

*Tài liệu chính của dự án: [`README.md`](README.md) · AI service: [`ai/README.md`](ai/README.md) ·
Kế hoạch tích hợp: [`plan.md`](plan.md)*
