---
title: eKYC AI Service
emoji: 🪪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# eKYC AI Service

FastAPI service cho xác minh giấy tờ (CCCD/CC): OCR, đối chiếu QR, so khớp
khuôn mặt, liveness, và phát hiện giả mạo tài liệu (tamper). Đây là REST API
để backend (NestJS) và frontend (Next.js) gọi vào.

> **Lưu ý:** File này là README cho **Hugging Face Space** (cần YAML front-matter
> để HF biết `sdk: docker` và `app_port: 7860`). Khi tạo Space, đổi tên file này
> thành `README.md` trong repo Space. Tài liệu kỹ thuật đầy đủ xem `ai/README.md`.

## Endpoints

| Method | Path | Mô tả |
| --- | --- | --- |
| GET | `/healthz` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/ocr/extract` | OCR ảnh giấy tờ |
| POST | `/qr/extract` | Giải mã QR trên mặt sau CCCD |
| POST | `/verify/` | Xác minh (multipart: `front`, `back?`, `selfie?`) |
| POST | `/tamper/analyze` | Phân tích giả mạo tài liệu |
| GET | `/tamper/healthz` | Trạng thái engine tamper |

## Biến môi trường

Cấu hình qua **Settings → Variables and secrets** của Space. Các biến chính:

| Biến | Mặc định | Mô tả |
| --- | --- | --- |
| `AI_TAMPER_ENABLED` | `true` | Bật kiểm tra giả mạo |
| `AI_TAMPER_MODEL` | `models/tamper_net.pth` | Checkpoint (`.pth` torch hoặc `.onnx`) |
| `AI_TAMPER_THRESHOLD` | `0.55` | Ngưỡng quyết định `TAMPERED` |
| `AI_TAMPER_SUSPECT` | `0.35` | Ngưỡng `SUSPICIOUS` |
| `AI_FAS_ENABLED` | `true` | Bật liveness passive (MiniFASNet) |
| `AI_QR_STRICT` | `true` | Reject khi số trên QR không khớp OCR |

Xem đầy đủ trong `ai/README.md`.

## Deploy

1. Tạo Space mới → chọn **SDK: Docker**.
2. Push thư mục `ai/` lên repo Space (Dockerfile, requirements-docker.txt,
   README.md, `app/`, `models/`). File này đổi tên thành `README.md`.
3. Space tự build; lần build đầu ~10–15 phút (torch CPU + mediapipe).
4. Sau khi chạy, API ở `https://<user>-<space>.hf.space`.

## Hardware

CPU basic (free) của HF: 2 vCPU / 16GB RAM — đủ cho torch + mediapipe + 5 model.
Không có GPU nên model chạy CPU (torch được cài bản CPU-only).
