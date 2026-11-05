FROM python:3.12-slim-bookworm@sha256:da54eb86849fe19be7584d8b3f291fcf0c53c196760ab3d70dc6382f681ef9c0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY coder/ coder/
COPY data/mms-2026-01.json data/mms-2026-01.json
ENTRYPOINT ["python", "-m", "coder"]
