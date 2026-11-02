#!/usr/bin/env python3
"""CLI script to set up and verify persistent local GGUF models in ./models/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphy.llm.model_setup import setup_models_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and verify GRAPHY local GGUF models.")
    parser.add_argument("--models-dir", default="models", help="Target models directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing models from HuggingFace if not found in local cache",
    )
    args = parser.parse_args()

    manifest = setup_models_volume(models_dir=args.models_dir, download_missing=args.download)
    print(json.dumps(manifest.model_dump(), indent=2))
    all_ready = all(e.ready for e in manifest.entries)
    return 0 if all_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
