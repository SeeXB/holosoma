#!/usr/bin/env python3
"""Run an upstream Python entry point with the cached Hugging Face token.

This wrapper only supplies authentication through the environment.  It does not
import or alter the entry point's implementation, arguments, or model settings.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_official_with_cached_hf_token.py ENTRYPOINT [ARGS ...]")

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Hugging Face token is empty: {token_path}")
    os.environ["HF_TOKEN"] = token
    # The host defaults to hf-mirror.com, which does not serve this gated
    # official checkpoint correctly.  Use the canonical Hub unless explicitly
    # overridden for this experiment.
    os.environ["HF_ENDPOINT"] = os.environ.get(
        "CARI4D_HF_ENDPOINT", "https://huggingface.co"
    )

    entrypoint = Path(sys.argv[1]).resolve()
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
