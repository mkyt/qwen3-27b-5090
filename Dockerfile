# syntax=docker/dockerfile:1
# The serving image: libq27.so built in the CUDA devel image, then plain Ubuntu with the CUDA runtime library
# (libcudart, the engine's only CUDA dependency; the driver comes from the NVIDIA container runtime), Python and the
# server's packages (frontend/requirements.txt). No compiler or PyTorch at runtime (dev/Dockerfile has those, for
# development). (On the nvidia/cuda base image it was 1.21 GB, 456 MB of it the forward-compatibility driver
# package, which GeForce GPUs cannot use.) Published as ghcr.io/mkyt/qwen3-27b-5090 (.github/workflows/image.yml); built here by
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

# Python and the packages, without uv itself in the image. --no-deps: requirements.txt lists every package the server
# imports (tokenizers' huggingface_hub, with httpx, hf_xet and others, is never used: the server reads tokenizer.json).
# The standalone Python's parts the server does not need go: the embedding library (the interpreter is static),
# Tcl/Tk, IDLE, ensurepip and pip, headers, manuals.
FROM ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c AS python
COPY --from=uv /uv /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
COPY frontend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.12 \
 && uv pip install --python /opt/venv --no-deps -r /tmp/requirements.txt \
 && cd "$(dirname "$(dirname "$(readlink -f /opt/venv/bin/python)")")" \
 && rm -rf lib/libpython3* lib/libtcl* lib/tcl* lib/tk* lib/itcl* lib/thread* lib/pkgconfig include share \
           bin/idle3* bin/pip* bin/2to3* bin/pydoc3* bin/python3*-config \
           lib/python3.12/idlelib lib/python3.12/tkinter lib/python3.12/turtledemo lib/python3.12/ensurepip \
           lib/python3.12/lib2to3 lib/python3.12/pydoc_data lib/python3.12/site-packages/pip* \
           lib/python3.12/lib-dynload/_tkinter* lib/python3.12/config-3.12*

FROM ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
COPY --from=python /opt/python /opt/python
COPY --from=python /opt/venv /opt/venv
COPY --from=build /usr/local/cuda/lib64/libcudart.so.13* /opt/q27/
# (what the nvidia/cuda images set for the container runtime: all GPUs, the compute and nvidia-smi libraries)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/opt/q27 \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
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
