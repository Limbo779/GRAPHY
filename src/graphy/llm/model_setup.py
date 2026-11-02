"""Model setup, GGUF header verification, and persistent volume manifest builder."""

from __future__ import annotations

import json
import os
import shutil
import struct
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class GGUFHeaderInfo(BaseModel):
    valid_magic: bool
    version: int
    tensor_count: int
    kv_count: int
    file_size_bytes: int
    file_size_gib: float


class ModelEntry(BaseModel):
    role: str
    filename: str
    sha256: str
    repo_id: str
    download_url: str
    header: Optional[GGUFHeaderInfo] = None
    ready: bool = False


class ModelManifest(BaseModel):
    models_dir: str
    entries: List[ModelEntry] = Field(default_factory=list)


DEFAULT_MODEL_SPECS: List[Dict[str, str]] = [
    {
        "role": "instruct_and_embedding",
        "filename": "qwen2-7b-instruct.gguf",
        "sha256": "3a4078d53b46f22989adbf998ce5a3fd090b6541f112d7e936eb4204a04100b1",
        "repo_id": "Qwen/Qwen2-7B-Instruct-GGUF",
        "download_url": "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf",
    },
    {
        "role": "vision_mmproj",
        "filename": "minicpm-v-mmproj.gguf",
        "sha256": "f8a805e9e62085805c69c427287acefc284932eb4abfe6e1b1ce431d27e2f4e0",
        "repo_id": "openbmb/MiniCPM-V-2_6-GGUF",
        "download_url": "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf",
    },
    {
        "role": "reasoning",
        "filename": "deepseek-r1-qwen3-8b.gguf",
        "sha256": "e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d",
        "repo_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B-GGUF",
        "download_url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
    },
]


def inspect_gguf_header(path: Path) -> GGUFHeaderInfo:
    """Read the 24-byte GGUF binary header (magic, version, tensor_count, kv_count)."""
    if not path.exists() or path.stat().st_size < 24:
        return GGUFHeaderInfo(
            valid_magic=False,
            version=0,
            tensor_count=0,
            kv_count=0,
            file_size_bytes=0,
            file_size_gib=0.0,
        )

    size_bytes = path.stat().st_size
    with path.open("rb") as f:
        header_bytes = f.read(24)

    magic = header_bytes[:4]
    if magic != b"GGUF":
        return GGUFHeaderInfo(
            valid_magic=False,
            version=0,
            tensor_count=0,
            kv_count=0,
            file_size_bytes=size_bytes,
            file_size_gib=round(size_bytes / (1024.0**3), 3),
        )

    version, tensor_count, kv_count = struct.unpack("<IQQ", header_bytes[4:24])
    return GGUFHeaderInfo(
        valid_magic=True,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        file_size_bytes=size_bytes,
        file_size_gib=round(size_bytes / (1024.0**3), 3),
    )


def setup_models_volume(
    models_dir: Path | str = "models",
    local_blob_dirs: Optional[List[Path]] = None,
    download_missing: bool = False,
) -> ModelManifest:
    """Populate `./models` via zero-copy hardlink from local blob cache or download."""
    target_dir = Path(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    search_dirs: List[Path] = []
    if local_blob_dirs:
        search_dirs.extend(local_blob_dirs)
    default_ollama_blobs = Path.home() / ".ollama" / "models" / "blobs"
    if default_ollama_blobs.is_dir():
        search_dirs.append(default_ollama_blobs)
    host_ollama_blobs = Path("/host_ollama_blobs")
    if host_ollama_blobs.is_dir():
        search_dirs.append(host_ollama_blobs)

    entries: List[ModelEntry] = []

    for spec in DEFAULT_MODEL_SPECS:
        dest_file = target_dir / spec["filename"]
        sha = spec["sha256"]

        if not dest_file.exists():
            # Check local content-addressable blob caches first
            for bdir in search_dirs:
                candidate = bdir / f"sha256-{sha}"
                if candidate.is_file():
                    try:
                        os.link(candidate, dest_file)
                    except OSError:
                        shutil.copy2(candidate, dest_file)
                    break

        if not dest_file.exists() and download_missing:
            urllib.request.urlretrieve(spec["download_url"], dest_file)

        hdr = inspect_gguf_header(dest_file) if dest_file.exists() else None
        ready = bool(hdr and hdr.valid_magic and hdr.file_size_bytes > 1024)

        entries.append(
            ModelEntry(
                role=spec["role"],
                filename=spec["filename"],
                sha256=sha,
                repo_id=spec["repo_id"],
                download_url=spec["download_url"],
                header=hdr,
                ready=ready,
            )
        )

    manifest = ModelManifest(models_dir=str(target_dir), entries=entries)
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    return manifest
