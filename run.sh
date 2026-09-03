#!/usr/bin/env bash
# piper-trainer launcher.
#
#   ./run.sh build                        # build the VARIANT image
#   ./run.sh build --no-cache             # extra args pass through to the build
#   ./run.sh doctor
#   ./run.sh init /workspace/marvin --name marvin
#   ./run.sh prepare /workspace/marvin --tier medium
#   ./run.sh serve                        # API + Bones UI on :8000
#
# Handles the runtime differences that compose files get wrong:
#   - rootless podman: no --user (host UID already maps to container root);
#     adding --user breaks writes to the mounted volume
#   - rootful docker:  --user is required or every file lands root-owned
#   - AMD:   --device /dev/kfd --device /dev/dri, plus supplementary groups
#            (podman resolves group NAMES inside the container, where Debian
#            has no 'render' group -> use keep-groups on podman, numeric GIDs
#            on docker)
#   - SELinux: :Z on the volume mount
#
# Env:
#   WORKSPACE  host directory to mount at /workspace   (default: ./workspace)
#   VARIANT    rocm | cuda | cpu                       (default: rocm)
#   IMAGE      override the full image reference
#   ENGINE     podman | docker                         (default: autodetect)
#   API_PORT   host port for `serve`                   (default: 8000)
#   ROCM_TORCH_VERSION   torch pin for `build`         (default: 2.10.0; must
#                        exist on the rolling gfx1151 index — see Dockerfile)
#   SHELL_IN   set to 1 to drop into a shell instead of running the CLI

set -euo pipefail

WORKSPACE="${WORKSPACE:-$PWD/workspace}"
VARIANT="${VARIANT:-rocm}"
IMAGE="${IMAGE:-piper-trainer:${VARIANT}}"

if [[ -z "${ENGINE:-}" ]]; then
    if command -v podman >/dev/null 2>&1; then ENGINE=podman
    elif command -v docker >/dev/null 2>&1; then ENGINE=docker
    else echo "need podman or docker on PATH" >&2; exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------- build
# `./run.sh build` maps VARIANT to the right index + pin. Only ROCm needs
# explicit build args: the Dockerfile defaults already match CUDA/CPU
# (2.6.0 on cu124/cpu), while the rolling gfx1151 nightly index carries
# different generations entirely (see the torch-layer NOTE in the
# Dockerfile). The pin here mirrors docker-compose.yml's rocm profile.
if [[ "${1:-}" == "build" ]]; then
    shift
    bargs=()
    if [[ "$VARIANT" == "rocm" ]]; then
        bargs+=(
            --build-arg "TORCH_INDEX_URL=${ROCM_TORCH_INDEX_URL:-https://rocm.nightlies.amd.com/v2/gfx1151/}"
            --build-arg "TORCH_VERSION=${ROCM_TORCH_VERSION:-2.10.0}")
    fi
    exec "$ENGINE" build "${bargs[@]}" -t "$IMAGE" "$@" "$SCRIPT_DIR"
fi

mkdir -p "$WORKSPACE"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"

args=(--rm -it)

# ---------------------------------------------------------------- user / uid
rootless=0
if [[ "$ENGINE" == "podman" ]]; then
    # `podman info` reports rootless directly; fall back to euid
    if podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null | grep -qi true; then
        rootless=1
    elif [[ "$(id -u)" != "0" ]]; then
        rootless=1
    fi
fi

if [[ "$rootless" == "1" ]]; then
    # Deliberately NO --user: host UID already maps to container root, so
    # files created on the volume come out owned by the invoking user.
    :
else
    args+=(--user "$(id -u):$(id -g)")
fi

# --------------------------------------------------------------- accelerator
case "$VARIANT" in
  rocm)
    args+=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined)
    if [[ "$ENGINE" == "podman" ]]; then
        args+=(--group-add keep-groups)
    else
        for g in video render; do
            gid="$(getent group "$g" | cut -d: -f3 || true)"
            [[ -n "$gid" ]] && args+=(--group-add "$gid")
        done
    fi
    ;;
  cuda)
    if [[ "$ENGINE" == "podman" ]]; then
        args+=(--device nvidia.com/gpu=all)
    else
        args+=(--gpus all)
    fi
    ;;
  cpu) ;;
  *) echo "unknown VARIANT: $VARIANT (rocm|cuda|cpu)" >&2; exit 1 ;;
esac

# -------------------------------------------------------------------- volume
mount_opts=""
if command -v getenforce >/dev/null 2>&1 && [[ "$(getenforce)" != "Disabled" ]]; then
    mount_opts=":Z"
fi
args+=(-v "${WORKSPACE}:/workspace${mount_opts}")

# training and Whisper both want more than the 64 MB default
args+=(--shm-size 8g)

# ----------------------------------------------------------------- serve
# `./run.sh serve` publishes API_PORT and defaults the container-internal
# listener to 0.0.0.0 (127.0.0.1 inside a container is unreachable from the
# host). An explicit --host on the command line always wins.
# The publish target on the HOST side defaults to 127.0.0.1: the API has no
# auth and can spawn jobs, so it is not offered to the LAN by default. Set
# API_BIND=0.0.0.0 to expose it to your network deliberately (design §7:
# the supported path is localhost / VPN).
if [[ "${1:-}" == "serve" ]]; then
    args+=(-p "${API_BIND:-127.0.0.1}:${API_PORT:-8000}:8000")
    host_set=0
    for a in "$@"; do
        case "$a" in --host|--host=*) host_set=1 ;; esac
    done
    [[ "$host_set" == "0" ]] && set -- "$@" --host 0.0.0.0
fi

if [[ "${SHELL_IN:-0}" == "1" ]]; then
    args+=(--entrypoint /bin/bash)
    exec "$ENGINE" run "${args[@]}" "$IMAGE"
fi

exec "$ENGINE" run "${args[@]}" "$IMAGE" "$@"
