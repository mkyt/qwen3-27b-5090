# syntax=docker/dockerfile:1
# The serving image: libq27.so built in the CUDA devel image, then only the CUDA runtime, Python and the server's
# four packages (frontend/requirements.txt). No compiler or PyTorch at runtime (dev/Dockerfile has those, for
# development). Published as ghcr.io/mkyt/qwen3-27b-5090 (.github/workflows/image.yml); built here by
# `docker compose up -d --build serve`.

FROM ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 AS uv

FROM nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
COPY engine /src/engine
# Parallel compile jobs (empty: one per CPU); the GitHub workflow limits them to fit the runner's RAM.
ARG BUILD_JOBS=
# The build directory stays in the BuildKit cache, so a change to one source recompiles only that file.
RUN --mount=type=cache,target=/build \
    cmake -S /src/engine -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null \
 && cmake --build /build --target q27 ${BUILD_JOBS:+-j $BUILD_JOBS} \
 && install -D /build/libq27.so /opt/q27/libq27.so

FROM nvidia/cuda:13.3.0-base-ubuntu24.04@sha256:bcf7d05f0b13b9bbb86d9a4cd039d331894b8f1145ad009d1af75023bcd1dc5c
COPY --from=uv /uv /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
COPY frontend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.12 \
 && uv pip install --python /opt/venv -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:${PATH} \
    Q27_LIB=/opt/q27/libq27.so \
    HF_HOME=/hf \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp
COPY --from=build /opt/q27/libq27.so /opt/q27/libq27.so
WORKDIR /work
COPY LICENSE ./
COPY bench/draft-vocab bench/draft-vocab
COPY reference/qwen27b/__init__.py reference/qwen27b/paths.py reference/qwen27b/image_tokens.py reference/qwen27b/
COPY engine/python engine/python
COPY frontend frontend
EXPOSE 8000
CMD ["python", "frontend/server.py"]
