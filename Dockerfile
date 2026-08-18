# syntax=docker/dockerfile:1
#
# piper-trainer — packaged piper1-gpl training pipeline
#
# Build (NVIDIA):
#   docker build -t piper-trainer:cuda .
#
# Build (AMD Strix Halo / gfx1151):
#   docker build -t piper-trainer:rocm \
#     --build-arg TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/ .
#
# Torch wheels from both indexes bundle their own accelerator runtime
# (nvidia-* packages / _rocm_sdk_libraries_gfx1151), so a plain Python base
# works for both. Only device passthrough differs at run time.

FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------- build args
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG PIPER_REF=main
ARG DEEPFILTER_VERSION=0.5.6
ARG WHISPER_MODEL=large-v3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------ system packages
# espeak-ng: phonemization (piper also vendors its own copy via CMake, but the
#            dev headers are needed to build espeakbridge)
# ffmpeg:    every audio conversion step
# cmake/ninja/build-essential: espeakbridge + monotonic_align
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libespeak-ng-dev \
        libsndfile1 \
        ninja-build \
        pkg-config \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------- deep-filter
# Static musl build: no glibc dependency, no Python, ~10 MB.
RUN curl -fsSL -o /usr/local/bin/deep-filter \
      "https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-x86_64-unknown-linux-musl" \
    && chmod +x /usr/local/bin/deep-filter \
    && deep-filter --version

# --------------------------------------------------------------------- torch
# Installed first and in its own layer: it is the largest and least volatile
# dependency, and the index URL is what distinguishes the CUDA/ROCm variants.
RUN python3 -m pip install --index-url "${TORCH_INDEX_URL}" torch torchaudio

# ----------------------------------------------------------------- piper1-gpl
WORKDIR /opt
RUN git clone --depth 1 --branch "${PIPER_REF}" \
      https://github.com/OHF-Voice/piper1-gpl.git /opt/piper1-gpl

WORKDIR /opt/piper1-gpl

# Build-time deps. NOTE: scikit-build (legacy), NOT scikit-build-core —
# setup.py imports `skbuild`.
RUN python3 -m pip install \
        scikit-build setuptools wheel cmake ninja cython onnxscript

# --- C extension 1: espeakbridge + vendored espeak-ng-data (CMake) -----------
# pyproject.toml declares build-backend = "setuptools.build_meta" while
# setup.py uses skbuild.setup(), so pip NEVER runs the CMake path. Building
# explicitly is mandatory; without it the install "succeeds" and then fails at
# runtime with ImportError: cannot import name 'espeakbridge'.
RUN python3 setup.py build_ext --inplace \
    && test -f src/piper/espeakbridge.so \
    && test -d src/piper/espeak-ng-data

# --- C extension 2: monotonic_align (Cython) --------------------------------
# The project's own script; it creates the required nested
# monotonic_align/monotonic_align/ package layout that the import path expects.
RUN ./build_monotonic_align.sh \
    && ls src/piper/train/vits/monotonic_align/monotonic_align/core*.so

# --- Patch: force the legacy TorchScript ONNX exporter -----------------------
# Recent torch defaults to the dynamo exporter, which cannot trace VITS's
# stochastic duration predictor (data-dependent assert in transforms.py) and
# dies with GuardOnDataDependentSymNode. Fail the build loudly if the upstream
# call signature changes rather than shipping a silently-broken export.
RUN grep -q "torch.onnx.export(" src/piper/train/export_onnx.py \
    && sed -i 's/torch\.onnx\.export(/torch.onnx.export(\n        dynamo=False,/' \
         src/piper/train/export_onnx.py \
    && grep -q "dynamo=False" src/piper/train/export_onnx.py

# --no-deps below protects the accelerator-specific torch build from being
# overwritten by the generic PyPI wheel — but it also skips everything else in
# the [train] extra, so those must be installed explicitly first.
RUN python3 -m pip install \
        "lightning>=2,<3" \
        "torchmetrics>0.7,<3" \
        "tensorboard>=2,<3" \
        "tensorboardX>=2,<3" \
        "jsonargparse[signatures]>=4.27.7" \
        "librosa<1" \
        "onnx>=1,<2" \
        "onnxruntime>=1,<2" \
        "pathvalidate>=3,<4" \
        "pysilero-vad>=2.1,<3"

RUN python3 -m pip install -e '.[train]' --no-build-isolation --no-deps

# ----------------------------------------------------- pipeline dependencies
RUN python3 -m pip install \
        "auditok>=0.3" \
        faster-whisper \
        soundfile \
        numpy

# --------------------------------------------------------- piper-trainer CLI
COPY pyproject.toml /opt/piper-trainer/pyproject.toml
COPY src /opt/piper-trainer/src
RUN python3 -m pip install -e /opt/piper-trainer --no-deps

# ------------------------------------------------------------------ runtime
# Everything writable lives under /workspace so the image runs fine as an
# arbitrary --user (no writable HOME required).
ENV HOME=/workspace \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HOME=/workspace/.cache/huggingface \
    PIPER_TRAINER_WHISPER_MODEL=${WHISPER_MODEL} \
    TORCH_BLAS_PREFER_HIPBLASLT=0

WORKDIR /workspace
VOLUME ["/workspace"]

# Smoke-test all three imports at build time. If any of these fail the image
# is broken and should not ship.
RUN python3 -c "from piper import espeakbridge" \
    && python3 -c "from piper.train.vits.monotonic_align.monotonic_align.core import maximum_path_c" \
    && python3 -c "import torch, auditok, faster_whisper, lightning, onnxscript" \
    && python3 -c "import piper.train.__main__"

ENTRYPOINT ["piper-trainer"]
CMD ["doctor"]
