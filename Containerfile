# =============================================================================
# GRAPHY & GRAPHY Companion — Rootless Podman Containerfile (Arch Linux Base)
# Optimized for 13th Gen Intel Core i5-1335U (AVX2/VNNI) + Intel Iris Xe (/dev/dri)
# =============================================================================
FROM docker.io/library/archlinux:latest

LABEL org.opencontainers.image.title="GRAPHY"
LABEL org.opencontainers.image.description="Self-discovery learning system for Math, Physics, and CS with GRAPHY Companion"
LABEL org.opencontainers.image.version="0.1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    LD_LIBRARY_PATH=/opt/llama:/opt/llama/vulkan \
    GRAPHY_CONFIG_PATH=/app/configs/config.yaml \
    PYTHONPATH=/app/src

# Install Python, Intel Iris Xe GPU (Vulkan + OpenCL + Level-Zero) drivers,
# BLAS/LAPACK/SQLite, and PDF/OCR tools (Poppler + Tesseract)
RUN pacman --cachedir /var/cache/pacman/pkg --cachedir /cache_extra -S --noconfirm --needed \
    python \
    python-cairo \
    python-pydantic \
    python-yaml \
    python-jinja \
    python-click \
    python-colorama \
    python-termcolor \
    python-requests \
    python-orjson \
    python-psutil \
    python-packaging \
    python-jsonschema \
    python-flask \
    python-flask-cors \
    python-waitress \
    python-markdown \
    python-beautifulsoup4 \
    poppler \
    poppler-data \
    tesseract \
    tesseract-data-eng \
    tesseract-data-osd \
    vulkan-icd-loader \
    vulkan-intel \
    vulkan-tools \
    intel-compute-runtime \
    intel-media-driver \
    clinfo \
    blas \
    cblas \
    lapack \
    sqlite

# Copy llama.cpp Vulkan + AlderLake AVX-VNNI + Multimodal runtime into /opt/llama
RUN mkdir -p /opt/llama && \
    if [ -d /ai_runtime/lib/ollama ]; then \
        cp -a /ai_runtime/lib/ollama/* /opt/llama/ && \
        ln -sf vulkan/libggml-vulkan.so /opt/llama/libggml-vulkan.so; \
    fi

WORKDIR /app

# Copy project source and configuration
COPY pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
COPY ideastorm.md /app/ideastorm.md
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY data/schemas/ /app/data/schemas/

# Create runtime directories for mounted persistent volumes
RUN mkdir -p /app/models \
             /app/data/raw \
             /app/data/processed \
             /app/data/vectors \
             /app/data/graphs \
             /app/data/scores \
             /app/data/uploads \
             /app/data/sessions

EXPOSE 8080

ENTRYPOINT ["python3", "-m", "graphy.cli.main"]
CMD ["hardware"]
