# syntax=docker/dockerfile:1
#
# piper-trainer — packaged piper1-gpl training pipeline
#
# Build (NVIDIA):
#   docker build -t piper-trainer:cuda .
#
# Build (AMD Strix Halo / gfx1151):
#   docker build -t piper-trainer:rocm \
#     --build-arg TORCH_INDEX_URL=https://rocm.nightlies.amd.com/v2/gfx1151/ \
#     --build-arg TORCH_VERSION=2.10.0 .
# (docker compose --profile rocm passes both args; a raw build must pass
#  them itself — the default TORCH_VERSION is the CUDA/CPU index's pin.)
#
# Torch wheels from both indexes bundle their own accelerator runtime
# (nvidia-* packages / _rocm_sdk_libraries_gfx1151), so a plain Python base
# works for both. Only device passthrough differs at run time.

# --------------------------------------------------------- React UI build
# Decision §8.4: React, with the build step in the image — annoying but
# bounded. ui-src/ builds to a static bundle that lands in
# src/piper_trainer/ui/ (the UI_DIR root) and rides the existing /ui
# StaticFiles mount as the only UI.
# This stage sits FIRST on purpose: the python stage below must stay
# contiguous — anything after a second FROM would execute in whatever
# stage that FROM declared.
FROM node:22-bookworm-slim AS ui-build
WORKDIR /ui
COPY ui-src/package.json ui-src/package-lock.json ./
RUN npm ci
COPY ui-src/ ./
# --outDir overrides vite.config.ts's repo-relative path (that one is for
# local `npm run build` writing straight into src/piper_trainer/ui/).
RUN npm run build -- --outDir /out --emptyOutDir

FROM python:3.12-slim-bookworm

# ---------------------------------------------------------------- build args
# Pinned by default so a --no-cache rebuild reproduces the same stack:
#   PIPER_REF      v1.7.0 tag (was: main, which floated across releases)
#   TORCH_VERSION  2.6.0, the version verified on the CUDA (cu124) and CPU
#                  indexes. ROCm builds MUST override it — see the note at
#                  the torch layer below.
# Override with --build-arg only when you deliberately want to move.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.6.0
ARG PIPER_REF=v1.7.0
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
# Exact pin first, prefix match second (the pytorch indexes use local
# version labels like 2.6.0+cu124). If neither resolves the build fails
# HERE rather than floating; the stamp + smoke test below re-assert it.
# NOTE: the pin is index-specific. The gfx1151 index is a ROLLING nightly
# that already dropped 2.6.0 (oldest live generation is 2.7.1), so ROCm
# builds pass --build-arg TORCH_VERSION=2.10.0 — the pin the rocm compose
# profile carries. Resolution relies on two PEP 440 wrinkles:
#   - torch==2.10.0 matches the index's 2.10.0+rocm* finals (a pin with
#     no local label ignores the candidate's);
#   - torchaudio only ships there as 2.10.0a0+rocm*, which ==2.10.0.*
#     picks up via the prefix fallback (pip allows prereleases when they
#     are the only match).
# When AMD rotates the index again the build fails HERE and pip's error
# lists the surviving generations — bump the pin deliberately.
RUN python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
    || python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}.*" "torchaudio==${TORCH_VERSION}.*" 

# --------------------------------------------------------------- piper1-gpl
# PIPER_REF is pinned (default v1.7.0) so a --no-cache rebuild reproduces
# the same upstream code; build with --build-arg PIPER_REF=<sha|tag> to
# move it deliberately.
WORKDIR /opt
RUN git clone --depth 1 --branch "${PIPER_REF}" \
      https://github.com/OHF-Voice/piper1-gpl.git /opt/piper1-gpl \
    && cd /opt/piper1-gpl && git rev-parse HEAD > /tmp/piper_sha

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
# inflect: spoken-form number expansion during transcribe (textnorm). Its
# absence does not fail anything — the fallback keeps digits — so it must
# ship here or fresh transcripts silently come out with "9000" unread.
RUN python3 -m pip install \
        "auditok>=0.3" \
        faster-whisper \
        "inflect>=7" \
        soundfile \
        numpy

# --------------------------------------------------- API server + Bones UI
# The [api] extra's runtime deps, installed explicitly because the editable
# install below uses --no-deps by design. websockets is not optional here:
# bare uvicorn refuses the WS upgrade ("Unsupported upgrade request") and
# /api/jobs/{id}/stream 404s, leaving the UI on its polling fallback.
# Bones (src/piper_trainer/ui/) needs no build step and ships with the
# COPY src above; the React bundle (built in ui-build, copied in below)
# lands in the same tree. Serve with ./run.sh serve (see run.sh for
# why not compose run under rootless podman).
RUN python3 -m pip install \
        "fastapi>=0.110" \
        "uvicorn>=0.29" \
        "websockets>=12" \
        "python-multipart>=0.0.9"

# ---------------------------------------------------- ingest source backends
# §2.5.5: media-site (yt-dlp) and hf-dataset (huggingface_hub) ingest guard
# on ImportError and degrade to a runtime error — which in a built image
# means the feature is a dead end nobody notices until the button fails.
# Ship both, pinned loosely; the smoke test below turns absence into a
# failed build instead of a broken button.
RUN python3 -m pip install \
        "yt-dlp>=2025.6.9" \
        "huggingface_hub>=0.20,<2"

EXPOSE 8000

# --------------------------------------------------------- piper-trainer CLI
COPY pyproject.toml /opt/piper-trainer/pyproject.toml
COPY src /opt/piper-trainer/src
COPY --from=ui-build /out/ /opt/piper-trainer/src/piper_trainer/ui/
RUN python3 -m pip install -e /opt/piper-trainer --no-deps

# ------------------------------------------------------------- versions stamp
# Written before the smoke tests; the torch assertion turns a silent float
# into a failed build. Read it with:
#   docker run --rm --entrypoint cat piper-trainer:cuda /opt/VERSIONS
# (Build args arrive as environment variables in RUN; the quoted heredoc
#  delimiter means no shell expansion happens inside the Python.)
RUN python3 - <<'EOF'
import os
import sys

import torch

installed = torch.__version__
pinned = os.environ.get("TORCH_VERSION", "")
if pinned and not installed.split("+")[0].startswith(pinned.split("+")[0]):
    print(f"torch {installed} does not match the pin {pinned!r} — the "
          f"index served a different version. Fix TORCH_VERSION or the "
          f"index URL; refusing to ship a float.", file=sys.stderr)
    sys.exit(1)
EOF
RUN piper_sha="$(cat /tmp/piper_sha)" \
    && printf 'piper1-gpl %s (ref %s)\ntorch %s (pin %s)\n' \
       "$piper_sha" "${PIPER_REF}" \
       "$(python3 -c 'import torch; print(torch.__version__)')" \
       "${TORCH_VERSION}" > /opt/VERSIONS \
    && cat /opt/VERSIONS

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
    && python3 -c "import yt_dlp, huggingface_hub" \
    && yt-dlp --version \
    && python3 -c "import piper.train.__main__" \
    && python3 -c "import piper_trainer.api.app" \
    && test -f /opt/piper-trainer/src/piper_trainer/ui/index.html \
    && test -n "$(ls /opt/piper-trainer/src/piper_trainer/ui/assets)"

ENTRYPOINT ["piper-trainer"]
CMD ["doctor"]
