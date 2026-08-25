#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# Run in the preprocessing environment chosen by the caller. For the default
# SAM3 backend, pass --sam3-python so SAM3 and CARI4D stay isolated.
PREPROCESS_PYTHON="${PREPROCESS_PYTHON:-python}"
exec "${PREPROCESS_PYTHON}" "${REPO_DIR}/tools/prepare_cari4d_video.py" "$@"
