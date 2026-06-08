# GPU-capable image. Base CUDA must be <= the host driver's CUDA (check `nvidia-smi`).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# Python 3.10 (Ubuntu 22.04 default; satisfies mineru >=3.10,<3.14) + system libs MinerU needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1 libglib2.0-0 poppler-utils fonts-noto-cjk \
    wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
