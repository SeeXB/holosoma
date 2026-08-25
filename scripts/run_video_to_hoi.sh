#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

# This wrapper intentionally uses the active CARI4D Python environment. It never installs packages
# and never enters the Isaac Sim environment.
exec python "${REPO_DIR}/tools/run_cari4d_custom.py" "$@"
