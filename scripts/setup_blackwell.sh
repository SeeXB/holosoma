#!/usr/bin/env bash
# Ubuntu 22.04+, x86_64, RTX 50-series / CUDA 12.8, Miniconda environments.
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/source_common.sh"
TARGET=${1:-all}
case "$TARGET" in all|training|retargeting) ;; *) echo "Usage: $0 [all|training|retargeting]" >&2; exit 2 ;; esac
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    echo "Install Miniconda at $CONDA_ROOT (or set CONDA_ROOT), then rerun this script." >&2
    exit 1
fi
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export OMNI_KIT_ACCEPT_EULA=1
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-300}
mkdir -p third_party outputs/environment-install/conda
# Prefer the previously downloaded official wheels when present.
wheel_options=()
if [[ -d "$ROOT_DIR/outputs/environment-install/wheels" ]]; then
    wheel_options+=(--find-links "$ROOT_DIR/outputs/environment-install/wheels")
fi

create_env() {
    local name=$1
    local env_path="$CONDA_ROOT/envs/$name"
    if [[ ! -d "$env_path/conda-meta" ]]; then
        "$CONDA_ROOT/bin/conda" create -y --prefix "$env_path" \
            --override-channels -c conda-forge python=3.11 pip
    fi
    source "$CONDA_ROOT/bin/activate" "$env_path"
    local python="$env_path/bin/python"
    "$python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
    # Complete tested snapshots are installed without re-resolving. This permits
    # the documented IsaacSim/Tyro typing-extensions pin conflict; check below
    # rejects every other dependency conflict. No vendor metadata is modified.
    "$python" -m pip install --no-deps --no-build-isolation \
        "${wheel_options[@]}" --extra-index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.nvidia.com \
        -r "$ROOT_DIR/scripts/requirements/$name-cu128.txt"
}

check_env() {
    local name=$1
    local python="$CONDA_ROOT/envs/$name/bin/python"
    "$python" - "$name" <<'CHECK'
import pathlib
import subprocess
import sys
name = sys.argv[1]
result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
output = result.stdout + result.stderr
print(output, end="")
pathlib.Path(f"outputs/environment-install/conda/{name}-pip-check.txt").write_text(output)
known_exception = (
    "isaacsim-kernel 5.1.0.0 has requirement typing-extensions==4.12.2, "
    "but you have typing-extensions 4.15.0."
)
if result.returncode:
    if name != "hssim" or output.strip().replace("typing_extensions", "typing-extensions") != known_exception:
        raise SystemExit(result.returncode)
    print("Known Isaac Sim / Tyro typing-extensions pin exception; see README.md.")
CHECK
    "$python" -m pip freeze > "outputs/environment-install/conda/$name-freeze.txt"
    "$CONDA_ROOT/bin/conda" env export --prefix "$CONDA_ROOT/envs/$name" \
        > "outputs/environment-install/conda/$name-environment.yml"
}

if [[ "$TARGET" == all || "$TARGET" == training ]]; then
    create_env hssim
    SIM_PY="$CONDA_ROOT/envs/hssim/bin/python"
    if [[ ! -d third_party/IsaacLab ]]; then
        git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
    fi
    sed -i 's/flatdict==4.0.1/flatdict==4.1.0/' third_party/IsaacLab/source/isaaclab/setup.py
    lab_packages=()
    for package in third_party/IsaacLab/source/*; do
        [[ ! -f "$package/setup.py" ]] || lab_packages+=(-e "$package")
    done
    "$SIM_PY" -m pip install --no-deps --no-build-isolation \
        "${lab_packages[@]}" -e 'src/holosoma[unitree,booster]'
    "$SIM_PY" - <<'FFMPEG'
import pathlib
import sys
import imageio_ffmpeg
link = pathlib.Path(sys.executable).parent / "ffmpeg"
if not link.exists():
    link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
FFMPEG
    check_env hssim
fi

if [[ "$TARGET" == all || "$TARGET" == retargeting ]]; then
    create_env hsretargeting
    "$CONDA_ROOT/envs/hsretargeting/bin/python" -m pip install --no-deps --no-build-isolation \
        -e 'src/holosoma_retargeting[dev]'
    check_env hsretargeting
fi
