#!/usr/bin/env python3
"""Run an upstream Python entry point against an explicit local HF cache."""

from __future__ import annotations

import argparse
import contextlib
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args, upstream_args = parser.parse_known_args()

    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/cari4d-matplotlib")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    entrypoint = args.entrypoint.resolve()
    sys.argv = [str(entrypoint), *upstream_args]
    if args.log is None:
        runpy.run_path(str(entrypoint), run_name="__main__")
        return

    args.log.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, value):
            for stream in self.streams:
                stream.write(value)
            return len(value)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    with args.log.open("w", encoding="utf-8") as log_stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_stream)):
            with contextlib.redirect_stderr(Tee(sys.stderr, log_stream)):
                runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
