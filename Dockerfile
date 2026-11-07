# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:da54eb86849fe19be7584d8b3f291fcf0c53c196760ab3d70dc6382f681ef9c0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app/src \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

COPY requirements.txt requirements-api.txt ./
RUN --mount=type=secret,id=system_ca \
    set -eu; \
    if [ -s /run/secrets/system_ca ]; then \
        export PIP_CERT=/run/secrets/system_ca \
               REQUESTS_CA_BUNDLE=/run/secrets/system_ca \
               SSL_CERT_FILE=/run/secrets/system_ca; \
    fi; \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch==2.9.1; \
    python -m pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Download the complete semantic/ folder before building; see instructions.md.
# Both query encoders and the fixed catalogue matrices are reused as-is.
COPY assets/semantic/ assets/semantic/
RUN set -eu; \
    for language in english multilingual; do \
        test -d "assets/semantic/$language/model"; \
        test -s "assets/semantic/$language/model/config.json"; \
        test -s "assets/semantic/$language/metadata.json"; \
        test -s "assets/semantic/$language/title_embeddings.npy"; \
    done

COPY src/ src/
COPY config/ config/
COPY scripts/finalize_tokenizers.py scripts/finalize_tokenizers.py
COPY data/mms-2026-01.json data/mms-2026-01.json
COPY run.py score.py ./
RUN python scripts/finalize_tokenizers.py

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python", "-m", "coder"]
