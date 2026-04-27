FROM python:3.11-slim

# faster-whisper depends on ctranslate2 which dlopens libssl + libcuda
# (when GPU). The slim base lacks libgomp / libstdc++ on some
# architectures — install the runtime libs whisper actually needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

# Install with the whisper extra so transcript.py can lazy-import
# faster-whisper without a separate sync step. This pulls
# ctranslate2 + huggingface_hub (~300 MB), but the streamer can
# still leave whisper_enabled=false and the dashboard starts fine.
RUN pip install --no-cache-dir -e ".[whisper]"

# Where faster-whisper caches downloaded models. Mount this from a
# named volume in compose so model files survive image rebuilds.
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["python", "-m", "chatterbot"]
CMD ["bot"]
