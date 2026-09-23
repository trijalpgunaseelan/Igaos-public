# syntax=docker/dockerfile:1
# IGAOS Solver Studio — solver + console in one image.
# Build context is the repository root:   docker build -f deploy/Dockerfile .

# ---------- stage 1: compile the solver ----------
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DIGAOS_BUILD_TESTS=OFF \
        -DIGAOS_BUILD_SHARED=OFF \
        -DIGAOS_HARDENING=ON \
 && cmake --build build -j"$(nproc)" --target igaos_cli \
 && strip build/igaos \
 && ./build/igaos --version

# ---------- stage 2: run it ----------
FROM python:3.12-slim AS run
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 igaos
WORKDIR /app
COPY --from=build /src/build/igaos  /app/build/igaos
COPY --from=build /src/demo         /app/demo
COPY --from=build /src/deploy       /app/deploy
ENV IGAOS_BIN=/app/build/igaos \
    PORT=8420 \
    IGAOS_HOST=0.0.0.0 \
    IGAOS_TIME_LIMIT=20 \
    IGAOS_MAX_CONCURRENT=2 \
    IGAOS_THREADS=1 \
    IGAOS_MAX_UPLOAD=2097152 \
    PYTHONUNBUFFERED=1
USER igaos
EXPOSE 8420
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8420')+'/healthz').read()"
CMD ["python3", "/app/deploy/serve.py"]
