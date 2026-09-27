# RTX 5090 environments

These are separate Miniconda environments under `~/miniconda3/envs/`.
`CONDA_ROOT` can override the Miniconda installation path.

| Component | Isaac Sim training | Motion retargeting |
| --- | --- | --- |
| Environment | `hssim` | `hsretargeting` |
| Python | 3.11 | 3.11 |
| PyTorch | 2.7.0+cu128 | 2.7.0+cu128 |
| NumPy | 1.26.0 | 2.3.5 |
| Simulator | Isaac Sim 5.1.0 / Isaac Lab 2.3.0 | MuJoCo |

The checked machine has two RTX 5090 GPUs, NVIDIA driver 590.48.01 and CUDA
Toolkit 12.8.61. `nvidia-smi` reports the driver's maximum supported CUDA version
(13.1); this does not change the installed toolkit or PyTorch's CUDA 12.8 runtime.

## Activate

From the repository root, use separate terminals or source the appropriate script:

```bash
source scripts/source_isaacsim_setup.sh
python -m pip show torch isaacsim isaaclab holosoma
```

```bash
source scripts/source_retargeting_setup.sh
python -m pip show torch mujoco holosoma-retargeting
```

## Reinstall

Requires Miniconda at `~/miniconda3` (or `CONDA_ROOT`), Git, Ubuntu 22.04+
(glibc 2.35+) and a working NVIDIA driver. Miniconda provides Python 3.11;
packages are installed using pip inside each Conda environment.

```bash
bash scripts/setup_blackwell.sh             # both environments
bash scripts/setup_blackwell.sh training    # training only
bash scripts/setup_blackwell.sh retargeting # retargeting only
```

The setup uses named Conda environments, keeps the system CUDA toolkit and driver,
installs Holosoma and Isaac Lab in editable mode, and provides an environment-local
FFmpeg executable. Isaac Lab is checked out at `third_party/IsaacLab` (tag v2.3.0;
commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`). The upstream `flatdict==4.0.1`
requirement is corrected to `4.1.0`, as in the original repository setup script.
The installation accepts the NVIDIA Omniverse EULA, matching the original setup.

Holosoma implements PPO/FastSAC itself. Its training setup includes Isaac Lab's
packages without the unrelated external RL framework extras.

## Dependency exception

Isaac Sim 5.1's `isaacsim-kernel` pins `typing-extensions==4.12.2`, but Holosoma's
required `tyro>=1.0.0` needs `typing-extensions>=4.13.0`. The complete package snapshot
`scripts/requirements/hssim-cu128.txt` selects `typing-extensions==4.15.0`.
The setup installs the full pinned dependency set with `pip --no-deps`, then
checks its consistency. Vendor metadata remains unchanged, so `pip check`
reports this known pin mismatch. The setup allows only this exact exception and
fails on other conflicts. Retargeting has no dependency exceptions.
AnyIO is pinned to 4.11.0 in training because newer releases require a newer
typing backport; this keeps that exception limited to Isaac Sim's vendor pin.

Do not upgrade PyTorch from the default PyPI index: the pinned CUDA 12.8 builds in
`scripts/requirements/*-cu128.txt` support the RTX 5090's `sm_120` architecture.
The training and retargeting packages require different NumPy major versions.

## Installation records

Local logs, smoke-test outputs and installed-package snapshots are under
`outputs/environment-install/conda/` (ignored by Git). GPU checks must run outside an
environment that hides `/dev/nvidia*`.

Verified on this machine on 2026-09-19:

- Both environments: CUDA matrix multiplication and backward propagation passed
  on both RTX 5090 cards (`sm_120`).
- Training: headless GPU rigid-body simulation passed; Holosoma G1 PPO completed
  2 iterations with 8 environments (384 timesteps), saved checkpoints and exited
  successfully. The standalone physics probe uses Holosoma's existing shutdown helper.
- Retargeting: MuJoCo model loading, CLARABEL optimization and a 4-frame OMOMO
  object-interaction retargeting run passed. The semantic runtime and input
  provenance tests passed (35 tests).
- Retargeting `pip check` is clean. Training reports only the documented
  `typing-extensions` vendor pin exception. The setup can reuse local wheels;
  `PIP_NO_INDEX=1` selects an offline reinstall when the complete wheel cache is present.

`verification.json`, `hssim-freeze.txt`, `hsretargeting-freeze.txt` and the Conda
exports in the installation-record directory contain the results and exact
installed versions.
Isaac Lab's release tag is v2.3.0; its individual Python packages have independent
versions (for example, the `isaaclab` distribution reports 0.47.2).

Official references:

- [Isaac Lab 2.3 pip installation](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- [PyTorch 2.7: Blackwell and CUDA 12.8 support](https://pytorch.org/blog/pytorch-2-7/)
- [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
