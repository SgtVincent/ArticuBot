#!/usr/bin/env bash
set -euo pipefail

# Start (or reuse) a GraspGen Docker container with ArticuBot-compatible mounts.
#
# Why this exists:
# - ArticuBot's on-demand grasp generation uses:
#     docker exec <container> python /code/scripts/generate_grasps_from_urdf.py --urdf-root <...>
# - We mount the ArticuBot repo at /workspace so GraspGen can read/write
#   assets under /workspace/data while keeping GraspGenModels (checkpoints)
#   available under /code/GraspGenModels.

usage() {
  cat <<'EOF'
Usage: scripts/run_graspgen_container.sh [options]

Options:
  --name NAME            Container name (default: friendly_galileo)
  --image IMAGE          Docker image (default: graspgen:latest)
  --code-dir PATH        Host GraspGen code dir mounted to /code (default: <repo>/third_party/GraspGen)
  --repo-dir PATH        Host repo dir mounted to /workspace (default: <repo>)
  --gpu-id N             Pin container to a single GPU id (uses --gpus "device=N")
  --no-gpu               Disable GPU usage (--gpus all omitted)
  --recreate             Remove any existing container with the same name and create a fresh one
  --attach               Attach an interactive shell after start
  -h, --help             Show this help

Examples:
  # Start the container (detached) for on-demand graspgen integration
  scripts/run_graspgen_container.sh

  # Attach a shell
  scripts/run_graspgen_container.sh --attach

  # After container is running, generate grasps for an object folder:
  docker exec -w /code friendly_galileo \
    python scripts/generate_grasps_from_urdf.py \
      --urdf-root /workspace/data/custom_objects/<your_object> \
      --gripper-config GraspGenModels/checkpoints/graspgen_franka_panda.yml
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="friendly_galileo"
IMAGE="graspgen:latest"
CODE_DIR="${REPO_ROOT}/third_party/GraspGen"
REPO_DIR="${REPO_ROOT}"
USE_GPU=1
GPU_ID=""
RECREATE=0
ATTACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --code-dir) CODE_DIR="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --gpu-id) GPU_ID="$2"; shift 2;;
    --no-gpu) USE_GPU=0; shift 1;;
    --recreate) RECREATE=1; shift 1;;
    --attach) ATTACH=1; shift 1;;
    -h|--help) usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

CODE_DIR="$(realpath "${CODE_DIR}")"
REPO_DIR="$(realpath "${REPO_DIR}")"

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "ERROR: code dir does not exist: ${CODE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "ERROR: repo dir does not exist: ${REPO_DIR}" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found in PATH" >&2
  exit 1
fi

if [[ ${RECREATE} -eq 1 ]]; then
  if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "Removing existing container: ${NAME}"
    docker rm -f "${NAME}" >/dev/null
  fi
fi

# If already running, just report and optionally attach.
if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "GraspGen container already running: ${NAME}"
  if [[ ${ATTACH} -eq 1 ]]; then
    exec docker exec -it -w /code "${NAME}" bash
  fi
  exit 0
fi

# If exists but stopped, start it.
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Starting existing container: ${NAME}"
  docker start "${NAME}" >/dev/null
  if [[ ${ATTACH} -eq 1 ]]; then
    exec docker exec -it -w /code "${NAME}" bash
  fi
  exit 0
fi

GPU_ARGS=()
if [[ ${USE_GPU} -eq 1 ]]; then
  if [[ -n "${GPU_ID}" ]]; then
    GPU_ARGS=(--gpus "device=${GPU_ID}" -e NVIDIA_DISABLE_REQUIRE=1 -e NVIDIA_DRIVER_CAPABILITIES=all)
  else
    GPU_ARGS=(--gpus all -e NVIDIA_DISABLE_REQUIRE=1 -e NVIDIA_DRIVER_CAPABILITIES=all)
  fi
fi

echo "Starting GraspGen container '${NAME}'"
echo "  Image:    ${IMAGE}"
echo "  /code <-  ${CODE_DIR}"
echo "  /workspace <- ${REPO_DIR}"

docker run \
  --privileged \
  --device /dev/dri \
  -itd \
  --name "${NAME}" \
  -v "${CODE_DIR}:/code" \
  -v "${REPO_DIR}:/workspace" \
  --net host \
  --shm-size 40G \
  "${GPU_ARGS[@]}" \
  "${IMAGE}" \
  /bin/bash -lc "cd /code && pip install -e . && bash"

echo "OK: container started: ${NAME}"
if [[ ${ATTACH} -eq 1 ]]; then
  exec docker exec -it -w /code "${NAME}" bash
fi
