# Stage 1: build the single-page app.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# Stage 2: print the PDF book.
#
# The book is built in the image so that a deploy is only a pull: the image
# carries the PDF that matches its own chapters, and no workstation has to build
# and copy one. Chromium, fonts, node_modules, and the loader's Python packages
# stay in this stage. The final image receives only the PDF and its manifest.
#
# This stage uses Debian rather than Alpine because Chromium and its fonts are
# better supported there, and because the print layout needs a node_modules
# built for glibc, which the Alpine `frontend` stage can't provide.
#
# Layers run from the least to the most frequently changed input, so a change
# to a backend file other than the loader's inputs reuses everything up to the
# last COPY. Don't pass GIT_SHA into this stage: a per-commit value would
# rebuild the book on every push. The manifest's commit is "unknown" for that
# reason.
FROM node:22-bookworm-slim AS book

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium python3 python3-venv fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_PATH=/usr/bin/chromium \
    PYTHON=/opt/venv/bin/python

WORKDIR /book

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN npm --prefix frontend ci --no-audit --no-fund

COPY backend/requirements.txt ./backend/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r backend/requirements.txt

# Only what the book is made from, as build-book.mjs hashes it.
COPY backend/app/__init__.py backend/app/config.py backend/app/curriculum.py ./backend/app/
COPY scripts/dump_book.py ./scripts/dump_book.py
COPY content/ ./content/
COPY frontend/book.html frontend/vite.book.config.ts frontend/tsconfig.json ./frontend/
COPY frontend/public/ ./frontend/public/
COPY frontend/src/ ./frontend/src/
COPY frontend/scripts/ ./frontend/scripts/
RUN node frontend/scripts/build-book.mjs --force --out /out \
    && rm -rf /out/.build

# Stage 3: the Python service, which also serves the built assets.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ ./backend/
COPY content/ ./content/
COPY engine/ ./engine/
COPY gpu/ ./gpu/
COPY --from=frontend /build/dist ./frontend/dist

# The commit this image was built from, reported by /api/health. Declared after
# the COPY steps so changing it does not invalidate the layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown -R app:app /data /app

# After the chown above, with its own --chown, so the PDF is stored once in
# the image and not a second time as a chown copy.
COPY --from=book --chown=app:app /out/build-an-inference-engine.pdf /out/manifest.json ./book/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
