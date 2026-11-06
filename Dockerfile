# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:da54eb86849fe19be7584d8b3f291fcf0c53c196760ab3d70dc6382f681ef9c0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt .
RUN --mount=type=secret,id=system_ca \
    set -eu; \
    if [ -s /run/secrets/system_ca ]; then \
        export PIP_CERT=/run/secrets/system_ca \
               REQUESTS_CA_BUNDLE=/run/secrets/system_ca \
               SSL_CERT_FILE=/run/secrets/system_ca; \
    fi; \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch==2.9.1; \
    python -m pip install --no-cache-dir -r requirements.txt

COPY scripts/build_semantic_assets.py scripts/build_semantic_assets.py
COPY data/mms-2026-01.json data/mms-2026-01.json

ARG ENGLISH_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
ARG ENGLISH_REVISION=090663c3ae57bf35ffe4d0d468a2a88d03051a4d
RUN --mount=type=secret,id=system_ca \
    set -eu; \
    if [ -s /run/secrets/system_ca ]; then \
        export REQUESTS_CA_BUNDLE=/run/secrets/system_ca \
               SSL_CERT_FILE=/run/secrets/system_ca; \
    fi; \
    HF_HOME=/tmp/hf-english python -m scripts.build_semantic_assets \
        --catalogue data/mms-2026-01.json \
        --output assets/semantic/english \
        --model "$ENGLISH_MODEL" --revision "$ENGLISH_REVISION" \
        --pooling cls --threads 4; \
    rm -rf /tmp/hf-english

ARG MULTILINGUAL_MODEL=cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR
ARG MULTILINGUAL_REVISION=47b6bd041ba61311584bb2494edfda5c7d9b719f
RUN --mount=type=secret,id=system_ca \
    set -eu; \
    if [ -s /run/secrets/system_ca ]; then \
        export REQUESTS_CA_BUNDLE=/run/secrets/system_ca \
               SSL_CERT_FILE=/run/secrets/system_ca; \
    fi; \
    HF_HOME=/tmp/hf-multilingual python -m scripts.build_semantic_assets \
        --catalogue data/mms-2026-01.json \
        --output assets/semantic/multilingual \
        --model "$MULTILINGUAL_MODEL" --revision "$MULTILINGUAL_REVISION" \
        --pooling cls --threads 4; \
    rm -rf /tmp/hf-multilingual

COPY scripts/finalize_tokenizers.py scripts/finalize_tokenizers.py
RUN python scripts/finalize_tokenizers.py

COPY coder/ coder/
ENTRYPOINT ["python", "-m", "coder"]
