# =============================================================================
# Parakeet TDT 0.6b v3 + Sortformer Diarization
# ARM64 / DGX Spark (CUDA 13, Blackwell GB10)
# Base: NVIDIA PyTorch NGC container (has ARM64 manifest)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:25.11-py3

LABEL maintainer="DGX Spark Voice Agent"
LABEL description="Parakeet TDT 0.6b v3 multilingual ASR + Sortformer speaker diarization with OpenAI-compatible API"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── 1. System dependencies ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Python dependencies ──────────────────────────────────────────────────
#    Install NeMo toolkit with ASR extras (includes diarization).
#    Pin lhotse to avoid the DynamicCutSampler TypeError that hits on newer versions.
#    The --break-system-packages flag is required in this container.
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR /opt/parakeet

COPY requirements.txt .

RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# ── 3. Pre-download models at build time (optional but saves startup time)
#    ASR model: ~1.2 GB, Diarization model: ~100 MB
#    Comment out if you prefer to download at first run.
# ─────────────────────────────────────────────────────────────────────────────
RUN python3 -c "\
import nemo.collections.asr as nemo_asr; \
print('Downloading parakeet-tdt-0.6b-v3...'); \
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3'); \
print('ASR model downloaded successfully'); \
del model \
"

RUN python3 -c "\
import nemo.collections.asr as nemo_asr; \
print('Downloading diar_sortformer_4spk-v1...'); \
model = nemo_asr.models.NeuralDiarizer.from_pretrained('nvidia/diar_sortformer_4spk-v1'); \
print('Diarization model downloaded successfully'); \
del model \
"

# ── 4. Copy application code ────────────────────────────────────────────────
COPY app/ /opt/parakeet/app/

# ── 5. Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── 6. Expose and run ───────────────────────────────────────────────────────
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
