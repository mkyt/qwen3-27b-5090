# syntax=docker/dockerfile:1
# Development image: CUDA toolkit (nvcc for SM120), CUTLASS headers, and a uv-managed Python
# environment for the PyTorch reference model, tests, quantization and bench clients.
# The serving engine image will be built separately from engine/ (no PyTorch at runtime).

FROM ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 AS uv

FROM nvidia/cuda:13.3.0-devel-ubuntu24.04@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52

RUN apt-get update \
 && apt-get install -y --no-install-recommends git cmake ninja-build ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# CUTLASS / CuTe headers (SM120 block-scaled MMA, TMA), pinned to the v4.8.0 commit.
ARG CUTLASS_COMMIT=098de2a652cf8f00fd70b2df54051c7eccbb855a
RUN git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass /opt/cutlass \
 && git -C /opt/cutlass checkout -q ${CUTLASS_COMMIT} \
 && rm -rf /opt/cutlass/.git
ENV CUTLASS_DIR=/opt/cutlass

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock /opt/dev/
# The download cache stays outside the image (BuildKit), so a change to the dependencies fetches only the difference.
RUN --mount=type=cache,target=/root/.cache/uv cd /opt/dev && uv sync --frozen --python 3.12
ENV PATH=/opt/venv/bin:${PATH} \
    HF_HUB_OFFLINE=1

WORKDIR /work
