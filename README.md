# Hệ thống xác thực CCCD (eKYC)

Hệ thống xác minh danh tính trực tuyến (eKYC) cho **CCCD/CMND Việt Nam**. Người dùng
chụp/upload hai mặt thẻ, quét khuôn mặt qua webcam để chứng minh người thật, hệ thống
tự động OCR, so khớp khuôn mặt, kiểm tra liveness và đối chiếu mã QR in trên thẻ, rồi trả
về kết quả **VERIFIED / REJECTED**.

Dự án gồm 3 tầng chạy độc lập:

| Tầng | Thư mục | Công nghệ | Cổng mặc định |
| --- | --- | --- | --- |
| Frontend | `fe/` | Next.js 15 (App Router) + React 19 | `3001` |
| Backend | `be/` | NestJS 10 + Prisma + PostgreSQL | `3000` |
| AI service | `ai/` | FastAPI + OCR/CV/Face/Anti-spoofing | `8001` |

Luồng tổng quát: **Next.js → NestJS → FastAPI AI → NestJS → PostgreSQL → Next.js**.
Frontend **không** gọi trực tiếp AI; mọi thứ đi qua NestJS để kiểm soát auth, lưu trữ và
trạng thái xác thực.

---

## Mục lục

- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt & chạy](#cài-đặt--chạy)
- [Biến môi trường](#biến-môi-trường)
- [Quy trình xác thực (flow)](#quy-trình-xác-thực-flow)
- [AI service chi tiết](#ai-service-chi-tiết)
  - [OCR](#1-ocr--trích-xuất-thông-tin)
  - [QR cross-check](#2-qr-cross-check)
  - [Face detection & matching](#3-face-detection--matching)
  - [Passive liveness](#4-passive-liveness-anti-spoofing)
  - [Active liveness](#5-active-liveness-thách-thức-quay-đầu)
  - [Quyết định cuối cùng](#6-quyết-định-cuối-cùng)
- [Backend API](#backend-api)
- [Cơ sở dữ liệu](#cơ-sở-dữ-liệu)
- [Frontend](#frontend)
- [Bảo mật & dữ liệu nhạy cảm](#bảo-mật--dữ-liệu-nhạy-cảm)
- [Kiểm thử](#kiểm-thử)
- [Xử lý sự cố](#xử-lý-sự-cố)

---

## Kiến trúc hệ thống

```text
┌──────────────────────────────┐
│   Next.js Web (fe/, :3001)   │
│   Login / KYC / Lịch sử      │
└──────────────┬───────────────┘
               │ REST (JSON + multipart/Bearer JWT)
               ▼
┌──────────────────────────────┐
│   NestJS API (be/, :3000)    │
│   Auth · Upload · Storage    │
│   Verification orchestration │
│   Swagger /api/docs          │
└───────┬───────────────┬──────┘
        │               │
        ▼               ▼
┌───────────────┐  ┌──────────────────────────────────────┐
│ PostgreSQL    │  │  FastAPI AI (ai/, :8001)             │
│ qua Prisma    │  │  PaddleOCR/RapidOCR · OpenCV         │
└───────────────┘  │  UltraFace · MediaPipe · ArcFace     │
                   │  MiniFASNetV2 · Active liveness      │
                   └──────────────────────────────────────┘
```

Nguyên tắc thiết kế:

- **Frontend chỉ gọi NestJS**, không gọi AI trực tiếp.
- **NestJS** chịu trách nhiệm: authentication, authorization, ownership, upload, lưu file,
  gọi AI, lưu kết quả, trả trạng thái.
- **AI service** chịu trách nhiệm: OCR, computer vision, face, liveness, ra quyết định AI.
- Dữ liệu nhạy cảm (ảnh CCCD/selfie) lưu local theo từng user, phục vụ qua endpoint có
  kiểm tra ownership, không phơi URL công khai.

---

## Cấu trúc thư mục

```text
IDVerification/
├── ai/                         # AI eKYC service (FastAPI)
│   ├── app/
│   │   ├── main.py             # Khởi tạo FastAPI + CORS + routers
│   │   ├── routers/            # health, ocr, qr, verify, liveness
│   │   └── services/           # ocr, qr, face, liveness, fasnet, active_liveness, verify
│   ├── models/                 # Các model ONNX/PTH/TASK (xem bên dưới)
│   ├── tests/                  # pytest cho từng module
│   └── requirements.txt
├── be/                         # NestJS backend
│   ├── prisma/
│   │   ├── schema.prisma       # User, IdentityVerification, VerificationQr
│   │   └── seed.ts             # Tạo user demo
│   ├── src/
│   │   ├── auth/               # Đăng ký / đăng nhập JWT
│   │   ├── ai/                 # Proxy gọi AI service
│   │   ├── storage/            # Lưu trữ file local theo user
│   │   ├── verification/       # Nghiệp vụ eKYC
│   │   └── prisma/             # PrismaService
│   ├── uploads/                # Ảnh đã upload (gitignored, trừ .gitkeep)
│   └── .env
├── fe/                         # Next.js frontend
│   ├── app/                    # page, login, register, kyc, verifications
│   ├── components/             # Nav, LivenessChallenge (webcam)
│   ├── lib/api.ts              # Client gọi REST API
│   └── .env.local
├── plan.md                     # Kế hoạch tích hợp AI eKYC
├── front.png / back.png        # Ảnh CCCD mẫu
└── selfie.png                  # Ảnh selfie mẫu
```

### Các model AI kèm theo (`ai/models/`)

| File | Kích thước | Vai trò |
| --- | --- | --- |
| `version-RFB-640.onnx` | ~1.5 MB | UltraFace face detector (input cố định 480×640) |
| `face_landmarker.task` | ~3.6 MB | MediaPipe FaceLandmarker (5 điểm landmark, blendshapes, ma trận pose) |
| `recognition.onnx` | ~13 MB | ArcFace embedding (112×112) để so khớp khuôn mặt |
| `2.7_80x80_MiniFASNetV2.pth` | ~1.8 MB | MiniFASNetV2 passive anti-spoofing (checkpoint PyTorch) |

---

## Yêu cầu môi trường

- **Node.js** ≥ 18 (khuyến nghị 20+) và npm
- **Python** ≥ 3.10 (khuyến nghị 3.11/3.12)
- **PostgreSQL** đang chạy (local hoặc Docker), tạo sẵn database
- **Webcam** và trình duyệt hỗ trợ `getUserMedia` cho bước liveness (Chrome/Edge/Firefox)
- (Tùy chọn) **GPU NVIDIA + CUDA** để tăng tốc OCR / anti-spoofing; mặc định chạy CPU

---

## Cài đặt & chạy

Khởi động theo thứ tự: **AI → BE → FE**. Mở 3 terminal riêng.

### 1. AI service (FastAPI, cổng 8001)

```bash
cd ai
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Kiểm tra: mở `http://127.0.0.1:8001/healthz` → `{"status":"ok"}` và
`http://127.0.0.1:8001/docs` để xem Swagger UI của AI.

> Lần chạy đầu tiên sẽ hơi lâu vì model/phụ thuộc (torch, paddleocr, mediapipe) được nạp.

### 2. Backend (NestJS, cổng 3000)

```bash
cd be
npm install
# Cấu hình .env (xem phần Biến môi trường) rồi:
npx prisma generate
npx prisma migrate dev      # hoặc: npx prisma db push
npm run seed                # tạo user demo@example.com / 123456
npm run start:dev
```

- API: `http://localhost:3000/api`
- Swagger: `http://localhost:3000/api/docs`
- Health: `http://localhost:3000/api/health`

### 3. Frontend (Next.js, cổng 3001)

```bash
cd fe
npm install
npm run dev
```

Mở `http://localhost:3001`. Đăng nhập bằng user demo (hoặc tự đăng ký) rồi vào **KYC**.

### Chạy nhanh bằng tài khoản demo

| Email | Mật khẩu |
| --- | --- |
| `demo@example.com` | `123456` |

> Nếu chưa muốn cài AI, có thể đặt `AI_MOCK=true` trong `be/.env` để backend trả kết quả
> `VERIFIED` giả lập (không gọi AI).

---

## Biến môi trường

### Backend — `be/.env`

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:1233@localhost:5432/cccd_ekyc_test?schema=public` | Chuỗi kết nối PostgreSQL |
| `JWT_SECRET` | `cccd-verification-dev-secret-change-me` | Khóa ký JWT (đổi khi production) |
| `JWT_EXPIRES_IN` | `1d` | Thời hạn token |
| `PORT` | `3000` | Cổng backend |
| `AI_SERVICE_URL` | `http://127.0.0.1:8001` | Địa chỉ AI service |
| `AI_SERVICE_TIMEOUT_MS` | `60000` | Timeout gọi AI (ms) |
| `UPLOAD_DIR` | `uploads` | Thư mục lưu ảnh |
| `AI_MOCK` | `false` | `true` để bỏ qua AI, luôn trả `VERIFIED` (dev/test) |

### Frontend — `fe/.env.local`

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:3000/api` | Base URL của backend |

### AI service — biến môi trường tùy chọn

**Passive liveness (MiniFASNetV2)**

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_FAS_ENABLED` | `true` | Bật engine MiniFASNetV2 |
| `AI_FAS_MODEL` | `models/2.7_80x80_MiniFASNetV2.pth` | Đường dẫn checkpoint |
| `AI_FAS_SCALE` | `2.7` | Hệ số mở rộng crop quanh face box |
| `AI_FAS_REAL_THRESHOLD` | `0.5` | Xác suất class "thật" tối thiểu để đạt |
| `AI_FAS_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `AI_FAS_TTA` | `true` | Average verdict qua lật ngang (test-time augmentation) |

**Active liveness (thách thức quay đầu)**

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_ACTIVE_STATIC_CHECK` | `true` | Từ chối khi các frame gần như không đổi (ảnh/điện thoại tĩnh) |
| `AI_ACTIVE_STATIC_DIFF` | `1.5` | Ngưỡng chênh lệch xám trung bình để coi là "tĩnh" |
| `AI_ACTIVE_REQUIRE_BLINK` | `false` | Bắt buộc phát hiện chớp mắt để đạt |

**QR cross-check**

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_QR_ENABLED` | `true` | Bật giải mã + so khớp QR |
| `AI_QR_REQUIRE` | `false` | Từ chối khi không tìm thấy QR đọc được |
| `AI_QR_STRICT` | `true` | Từ chối khi số CCCD trên QR không xuất hiện trong OCR |
| `AI_QR_STRICT_NAME` | `false` | Từ chối khi họ tên trên QR không khớp OCR |

**Face matching**

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `AI_MATCH_SCORE` | `0.30` | Ngưỡng cosine similarity để coi là MATCH |
| `AI_PARTIAL_SCORE` | `0.20` | Ngưỡng để coi là PARTIAL |

---

## Quy trình xác thực (flow)

```text
1. User đăng nhập  →  nhận JWT
2. Mở trang KYC
3. Upload CCCD mặt trước (bắt buộc) + mặt sau (tùy chọn)
4. Bật webcam → thử thách liveness: nhìn thẳng → quay trái → quay phải
   (frame "nhìn thẳng" đầu tiên được dùng làm selfie để so khớp khuôn mặt)
5. Submit multipart (front, back?, selfie, liveness[])
6. NestJS: validate + lưu file theo user + tạo record (PENDING → PROCESSING)
7. NestJS gọi FastAPI /verify/
8. AI chạy: OCR → QR cross-check → document quality → face detection →
   face embedding + matching → passive liveness → active liveness → decision
9. NestJS map status, kiểm tra CCCD trùng, lưu kết quả + bảng VerificationQr
10. Trả về VERIFIED / REJECTED (hoặc ERROR) cho frontend
```

---

## AI service chi tiết

FastAPI app (`ai/app/main.py`) mount 5 router: `health`, `ocr`, `qr`, `verify`, `liveness`.
Các suy luận nặng chạy trong **threadpool** (`run_in_threadpool`) để endpoint không bị block.

### Endpoints

| Method | Path | Mô tả |
| --- | --- | --- |
| GET | `/healthz` | Health check |
| POST | `/ocr/extract` | Upload 1 ảnh → trả OCR lines + full text + document_number |
| POST | `/qr/extract` | Upload 1 ảnh → giải mã QR (payload thô + field đã parse) |
| POST | `/verify/` | Multipart `front`, `back?`, `selfie?`, `liveness[]` → kết quả xác thực |
| GET | `/liveness/healthz` | Health check liveness |
| POST | `/liveness/pose` | Ước lượng head pose cho 1 frame (dùng cho hướng dẫn realtime) |
| POST | `/liveness/active` | Kiểm tra chuỗi thách thức (thẳng + trái + phải) |

### 1. OCR — trích xuất thông tin

`ai/app/services/ocr_service.py`

- Engine: **RapidOCR** (ONNX Runtime, ưu tiên vì ổn định), fallback **PaddleOCR** (`lang="vi"`).
- Ảnh được **EXIF-rotate** khi nạp → ảnh chụp thẳng từ điện thoại vẫn xử lý đúng.
- Output: danh sách `lines` (`{text, score}`), `full_text`, và `document_number`.
- `_find_document_number`: ưu tiên CCCD **12 chữ số**, rồi CMND cũ **9 chữ số**, tránh nhầm ngày tháng (8 chữ số).

### 2. QR cross-check

`ai/app/services/qr_service.py`

- Giải mã QR mặt sau thẻ bằng **OpenCV `QRCodeDetector`** (+ `QRCodeDetectorAruco`), fallback **pyzbar**.
- Thử nhiều biến thể tiền xử lý (resize, grayscale, upscale, adaptive threshold) để tăng tỉ lệ đọc được.
- Payload CCCD dạng `|` phân tách:
  `<document_number>|<legacy_id?>|<full_name>|<dob>|<gender>|<native_place>|<issue_date>`.
  Parser **chịu được việc thiếu field** (phân loại theo hình dạng, không theo vị trí cố định).
- `compare()` đối chiếu QR với text OCR → verdict: `NOT_FOUND` / `MATCH` / `PARTIAL` / `MISMATCH`.
  - Lưu ý: QR **không phải bằng chứng mật mã** (có thể photocopy), nên chỉ là tín hiệu chống giả mạo.
- Nếu OCR không đọc được số CCCD, số từ QR được dùng làm fallback.

### 3. Face detection & matching

`ai/app/services/face_service.py`

- **Detector**: UltraFace `version-RFB-640.onnx`, input cố định **480×640 (H×W)**, tiền xử lý BGR `(x-127.5)/127.5`. Nếu model lỗi → fallback Haar Cascade.
- **Landmarks**: MediaPipe FaceLandmarker (`face_landmarker.task`) → 5 điểm (mắt trái, mắt phải, mũi, khóe miệng trái/phải).
- **Alignment**: `cv2.estimateAffinePartial2D` về template ArcFace 112×112 chuẩn.
- **Recognition**: `recognition.onnx` (ArcFace) → embedding 112×112 chuẩn hóa `(x-127.5)/127.5`, chuẩn hóa L2.
- Với ảnh lớn (thẻ CCCD độ phân giải cao), ảnh được **chia tile chồng lấn** để tìm mặt in nhỏ.
- So khớp bằng **cosine similarity** trên embedding đã chuẩn hóa:
  - `≥ 0.30` (env `AI_MATCH_SCORE`) → `MATCH`
  - `≥ 0.20` → `PARTIAL`
  - còn lại → `NO_MATCH`
- Selfie phải chứa **đúng 1 khuôn mặt**; nhiều mặt bị từ chối.

### 4. Passive liveness (anti-spoofing)

`ai/app/services/liveness_service.py` + `ai/app/services/fasnet.py`

- Engine chính: **MiniFASNetV2** (`2.7_80x80_MiniFASNetV2.pth`), 3 class; index 1 = thật, 0/2 = giả (print/replay).
- Kiểm tra **texture** trên crop 80×80 mở rộng ×2.7 quanh face box → phân biệt mặt thật với ảnh in / màn hình replay.
- Tiền xử lý **phải khớp reference**: crop đưa vào ở **BGR trong [0, 255]** (bỏ bước `/255`; nếu chia về [0,1] thì mọi mặt thật đều bị coi là giả).
- TTA (lật ngang) giúp verdict ổn định hơn.
- Fallback khi thiếu torch/checkpoint hoặc lỗi suy luận: kiểm tra **Laplacian variance** (ngưỡng blur `25.0`).
- **Hạn chế**: không phát hiện được **mask 3D / mannequin**, và không phát hiện **video replay**.

### 5. Active liveness (thách thức quay đầu)

`ai/app/services/active_liveness_service.py`

- Người dùng được yêu cầu: **nhìn thẳng → quay trái → quay phải**. Hệ thống ước lượng **yaw** từng frame.
- Ngưỡng: `|yaw| ≤ 12°` = nhìn thẳng; `|yaw| ≥ 18°` = đã quay; cần mặt trong `≥ 80%` số frame; biên độ yaw `≥ 18°`.
- Chống ảnh/điện thoại tĩnh: so sánh các frame gần như giống hệt (mean grayscale diff < `AI_ACTIVE_STATIC_DIFF`).
- Tín hiệu **chớp mắt** từ blendshapes (`AI_ACTIVE_REQUIRE_BLINK` để bắt buộc).
- Ảnh tĩnh hoặc screen replay không thể tạo hai hướng quay đầu ngược nhau → bị từ chối.

### 6. Quyết định cuối cùng

`ai/app/services/verify_service.py` — tổ hợp tất cả tín hiệu:

```text
VERIFIED  khi: face_match == MATCH
              AND passive_liveness.is_live == true
              AND active_liveness.status == PASS
              AND không có QR reject
              AND không có nhiều khuôn mặt trong selfie

REJECTED  khi: bất kỳ điều kiện trên sai
               (face mismatch/weak, liveness fail, no face, multiple faces, QR mismatch)

PROCESSING: trạng thái trung gian mặc định
```

Kết quả trả về gồm: `status`, `ocr`, `qr`, `qr_consistency`, `document_quality`,
`face`, `face_match`, `selfie_face_count`, `multiple_faces`, `passive_liveness`,
`active_liveness`, `rejection_reason`.

---

## Backend API

Base URL: `http://localhost:3000/api` · Swagger: `/api/docs`

### Auth (`be/src/auth`)

| Method | Endpoint | Body | Mô tả |
| --- | --- | --- | --- |
| POST | `/api/auth/register` | `{email, password, fullName?}` | Đăng ký, trả `{user, accessToken}` |
| POST | `/api/auth/login` | `{email, password}` | Đăng nhập, trả `{user, accessToken}` |
| GET | `/api/auth/me` | — (Bearer) | Thông tin user hiện tại |

- Mật khẩu hash bằng **bcrypt** (10 rounds). JWT chứa `sub` + `email`, hết hạn theo `JWT_EXPIRES_IN`.
- `JwtStrategy.validate` nạp user từ DB; user không tồn tại → `401`.

### AI proxy (`be/src/ai`)

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| GET | `/api/ai/health` | Kiểm tra AI service có reachable không |
| POST | `/api/ai/liveness/pose` | Multipart `file` → head pose 1 frame (dùng cho UI realtime) |

### Verifications (`be/src/verification`)

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| POST | `/api/verifications` | Multipart: `front` (bắt buộc), `back?`, `selfie?`, `liveness[]`, `documentType?` → tạo + xử lý |
| GET | `/api/verifications` | Danh sách của user (lọc `?status=`, `?documentNumber=`) |
| GET | `/api/verifications/:id` | Chi tiết 1 verification (kèm QR) |
| GET | `/api/verifications/:id/image/:side` | Stream ảnh (`front`/`back`/`selfie`), có kiểm tra ownership |
| POST | `/api/verifications/:id/retry` | Thử lại, có thể thay ảnh mới |

Chi tiết triển khai:

- **Upload**: `FileFieldsInterceptor`; giới hạn mỗi file **8 MB**, tối đa 40 frame liveness.
- **Xử lý**: tạo record `PENDING` → `PROCESSING` → gọi AI → map status → lưu JSON kết quả.
- **Chống trùng CCCD**: nếu `documentNumber` đã có ở một verification `VERIFIED` khác →
  từ chối với lý do `CCCD ... is already verified`.
- **Ownership**: mọi thao tác `findOneForUser`/`getPreviewMeta`/`retry` đều so `userId`;
  truy cập record của người khác → `403 Forbidden`.
- **Lỗi AI**: `ERROR` + `rejectionReason: "AI error: ..."`; timeout → `504`, không reachable → `503`.
  Tuyệt đối không trả `VERIFIED` khi AI lỗi.

### Storage (`be/src/storage`)

- Lưu file vào `UPLOAD_DIR/<userId>/<uuid>.<ext>`.
- Chỉ chấp nhận MIME: `image/jpeg`, `image/jpg`, `image/png`, `image/webp`.
- Tên file random UUID → không đoán được, không phơi thư mục public.

---

## Cơ sở dữ liệu

`be/prisma/schema.prisma` — PostgreSQL qua Prisma.

### Enum `VerificationStatus`
`PENDING` · `PROCESSING` · `VERIFIED` · `REJECTED` · `ERROR`

### `User`
| Field | Kiểu | Ghi chú |
| --- | --- | --- |
| `id` | String | `cuid()` |
| `email` | String | `@unique` |
| `password` | String | bcrypt hash |
| `fullName` | String? | |
| `role` | String | mặc định `USER` |
| `verifications` | IdentityVerification[] | quan hệ 1-n |
| `createdAt` / `updatedAt` | DateTime | |

### `IdentityVerification`
| Field | Kiểu | Ghi chú |
| --- | --- | --- |
| `id` | String | `cuid()` |
| `userId` | String | FK → User (cascade delete) |
| `documentType` | String | mặc định `CCCD` |
| `documentNumber` | String? | số đọc được (index) |
| `fullNameOnDoc` | String? | họ tên lấy từ QR |
| `frontImagePath` / `backImagePath?` / `selfiePath?` | String | tên file lưu |
| `status` | VerificationStatus | mặc định `PENDING` (index) |
| `ocrResult` / `documentQuality` / `faceMatchResult` / `livenessResult` / `aiRawResult` | Json? | từng phần & raw AI |
| `qr` | VerificationQr? | quan hệ 1-1 |
| `rejectionReason` | String? | lý do từ chối |
| `createdAt` / `updatedAt` | DateTime | |

### `VerificationQr`
Lưu dữ liệu QR đã giải mã (1 dòng / verification) để audit độc lập với JSON AI:
`found`, `reader`, `side`, `payload`, `documentNumber`, `legacyId`, `fullName`, `dob`,
`gender`, `nativePlace`, `issueDate`, `consistencyStatus`, `consistencyScore`,
`matchedFields[]`, `mismatchedFields[]`, `raw`.

Các lệnh Prisma: `npm run prisma:generate` · `prisma:migrate` · `prisma:push` · `prisma:studio` · `seed`.

---

## Frontend

Next.js 15 App Router + React 19, giao diện tối (CSS thuần trong `app/globals.css`).

| Route | File | Chức năng |
| --- | --- | --- |
| `/` | `app/page.tsx` | Trang giới thiệu + link nhanh |
| `/login` | `app/login/page.tsx` | Đăng nhập (mặc định `demo@example.com` / `123456`) |
| `/register` | `app/register/page.tsx` | Đăng ký tài khoản mới |
| `/kyc` | `app/kyc/page.tsx` | Upload 2 mặt CCCD + liveness + submit |
| `/verifications` | `app/verifications/page.tsx` | Lịch sử & chi tiết (ảnh, QR, raw AI) |

- **Xác thực**: JWT lưu ở `localStorage` (`cccd_token`), đính kèm `Authorization: Bearer` trong `lib/api.ts`.
- **Bước KYC**: upload front/back (có preview), rồi component `LivenessChallenge`.
- **`components/LivenessChallenge.tsx`**: mở webcam qua `getUserMedia`, mỗi `550ms` chụp 1 frame
  (320×240 JPEG) gửi `/api/ai/liveness/pose`; hướng dẫn người dùng lần lượt theo phase
  (nhìn thẳng → quay trái → quay phải), thu các frame và gọi `onComplete(frames)`.
  Ngưỡng trên UI trùng với AI: `STRAIGHT_MAX = 12`, `TURN_MIN = 18`.
- Frame "nhìn thẳng" đầu tiên được dùng làm **selfie** để so khớp với ảnh CCCD.
- Ảnh lịch sử tải qua endpoint có auth rồi tạo `ObjectURL` (không dùng URL trực tiếp).

---

## Bảo mật & dữ liệu nhạy cảm

CCCD/selfie là dữ liệu cá nhân nhạy cảm, vì vậy:

- **Authentication & Authorization**: mọi endpoint eKYC yêu cầu JWT; kiểm tra **ownership**
  (user A không đọc được verification của user B — `403`).
- **Validate file**: kiểm MIME, kích thước (≤ 8 MB), tên file random UUID.
- **Không public ảnh**: file nằm ngoài web root, chỉ phục vụ qua endpoint có auth + `Cache-Control: private, no-store`.
- **Không để rò rỉ trạng thái**: AI lỗi → `ERROR`/`503`/`504`, không bao giờ `VERIFIED`.
- **Logging**: tránh log số CCCD đầy đủ, OCR full, ảnh raw nếu không cần thiết.
- **Webcam**: chỉ thu frame ngắn hạn cho liveness, không lưu video lâu dài.

> `JWT_SECRET`, `DATABASE_URL` và mọi `.env` **không** được commit (đã có trong `.gitignore`).
> Nhớ đổi `JWT_SECRET` mặc định khi triển khai thật.

---

## Kiểm thử

### AI (pytest)

```bash
cd ai
python -m pytest -v
# Riêng từng module:
python -m pytest tests/test_liveness.py -v
python -m pytest tests/test_qr_service.py -v
```

Bộ test bao phủ: crop geometry + regression input range [0,255] của MiniFASNet,
decode QR + cross-check OCR, parse số CCCD, tiền/hậu xử lý UltraFace, active liveness
(no-frame, frozen frames). Test tự **skip** nếu thiếu torch/checkpoint.

Tạo ảnh QR test:

```bash
pip install "qrcode[pil]"
python tests/make_qr_fixture.py "068096013455|251043755|Trần Chí Hào|08101996|Nam|Hà Nội|30082021" back_qr.png
```

### Backend (Jest)

```bash
cd be
npm test          # unit
npm run test:e2e  # e2e (nếu cấu hình)
```

---

## Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
| --- | --- | --- |
| `AI service unavailable` / `503` | AI chưa chạy, sai `AI_SERVICE_URL` | Bật AI ở cổng 8001; kiểm `/api/ai/health` |
| AI timeout `504` | Model nạp lần đầu chậm / máy yếu | Tăng `AI_SERVICE_TIMEOUT_MS`, chờ warm-up |
| Mọi mặt thật bị coi là giả | Tiền xử lý FAS sai range | Không chia `/255`; input phải ở [0,255] BGR |
| Không mở được camera | Thiếu quyền / không HTTPS | Cấp quyền webcam; `localhost` được coi là secure |
| Face luôn `NO_MATCH` | Ngưỡng quá cao / ảnh mờ | Hạ `AI_MATCH_SCORE`, dùng ảnh rõ hơn |
| OCR ra rỗng | Ảnh mờ/tối/nghiêng | Chụp lại đủ sáng, thẳng, nét |
| QR không đọc được | Ảnh mờ, QR bị che | Chụp rõ mặt sau; hoặc bật `AI_QR_REQUIRE=false` |
| `front image is required` | Thiếu ảnh mặt trước | Đảm bảo field `front` trong multipart |
| Prisma lỗi kết nối | Sai `DATABASE_URL` / DB chưa chạy | Kiểm PostgreSQL + chạy `prisma migrate` |
| Muốn test không cần AI | — | Đặt `AI_MOCK=true` trong `be/.env` |

---

## Ghi chú kỹ thuật

- Ảnh được **EXIF-rotate** ở mọi chỗ nạp → ảnh từ điện thoại không bị quay sai.
- Suy luận AI chạy trong **threadpool** giữ FastAPI phản hồi nhanh.
- Các engine đều có **fallback** để service vẫn sống khi thiếu model/phụ thuộc:
  OCR (Paddle→Rapid đảo ngược tùy môi trường), detector (UltraFace→Haar),
  passive liveness (MiniFASNet→Laplacian blur).
- Tài liệu gốc của AI service: xem [`ai/README.md`](ai/README.md).
- Kế hoạch tích hợp chi tiết: xem [`plan.md`](plan.md).
