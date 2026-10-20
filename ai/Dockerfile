# Dockerfile cho AI eKYC service - Hugging Face Docker Space (CPU, 16GB RAM free).
#
# HF Docker Space đọc port từ YAML ở đầu README.md (sdk: docker, app_port: 7860),
# KHÔNG đọc EXPOSE/LABEL. Vì vậy app phải listen đúng 7860.
#
# Chạy local để test:
#   cd ai
#   docker build -t ekyc-ai .
#   docker run -p 7860:7860 ekyc-ai
#
# Deploy lên HF: tạo Space -> SDK = Docker. Push thư mục ai/ lên Space repo
# (Dockerfile này, requirements-docker.txt, README.md, app/, models/).
FROM python:3.11-slim-bookworm

# Thư viện hệ thống tối thiểu cho opencv/mediapipe/onnxruntime.
# libgl1 + libglib2.0-0: opencv cần dù là bản headless.
# libgomp1: OpenMP cho onnxruntime/torch.
# ffmpeg/libsm6/libxext6: mediapipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        ffmpeg \
        libsm6 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# 1) torch CPU-only trước (tránh kéo bản CUDA ~2.5GB; HF Space không có GPU).
ARG TORCH_VERSION=2.13.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}"

# 2) Runtime deps còn lại (RapidOCR thay PaddleOCR cho nhẹ).
COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install -r requirements-docker.txt

# 3) Code + model weights (models/ được track trong git, ~22MB).
COPY app ./app
COPY models ./models

# --- HF Space yêu cầu user uid 1000 để Dev Mode hoạt động -------------------
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

# Healthcheck dựa trên endpoint /healthz sẵn có của service.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/healthz || exit 1

# 1 worker để không nhân đôi RAM model. HF proxy cần bind 0.0.0.0:7860.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
