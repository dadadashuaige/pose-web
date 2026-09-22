# 默认构建 CPU 版镜像：体积小、不依赖 nvidia runtime，clone 下来就能跑。
# 需要 GPU 时把基础镜像换成 nvidia/cuda 并安装对应 CUDA 版 torch。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/backend/data/ultralytics

# OpenCV / reportlab 运行所需的系统库；fonts-noto-cjk 保证 PDF 中文不出现方框
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
# 先装 CPU 版 torch，避免 ultralytics 拉入体积巨大的 CUDA 轮子
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r backend/requirements.txt

COPY backend ./backend
RUN mkdir -p /app/backend/data

# 必须切到 backend/：配置里的 ./data 是相对当前工作目录解析的，
# 而且 uvicorn 要能以 app.main:app 的形式导入（app 包就在 backend/ 下）。
WORKDIR /app/backend

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
