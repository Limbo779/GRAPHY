"""Pydantic-validated configuration loader for GRAPHY."""

from __future__ import annotations

from pathlib import Path
from typing import List
import os
import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    project_name: str = "GRAPHY"
    version: str = "0.1.0"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")


class GPUConfig(BaseModel):
    enabled: bool = True
    vendor_id: str = "0x8086"
    device_name: str = "Intel Iris Xe Graphics"
    dri_render_node: str = "/dev/dri/renderD128"
    backend_preference: List[str] = Field(
        default_factory=lambda: ["vulkan", "sycl", "openvino", "cpu_avx2_vnni"]
    )


class HardwareConfig(BaseModel):
    target_cpu_arch: str = "x86_64"
    total_logical_cpus: int = 12
    physical_cores: int = 10
    p_core_threads: List[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    p_core_physical: List[int] = Field(default_factory=lambda: [0, 2])
    e_core_physical: List[int] = Field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9, 10, 11]
    )
    llm_generation_threads: int = 8
    llm_batch_threads: int = 10
    embedding_threads: int = 8
    max_container_ram_gib: float = 6.5
    min_free_ram_for_llm_gib: float = 3.0
    gpu: GPUConfig = Field(default_factory=GPUConfig)


class LLMModelConfig(BaseModel):
    repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    context_window: int = 4096
    gpu_layers: int = 20
    temperature: float = 0.1


class EmbeddingModelConfig(BaseModel):
    repo_id: str = "BAAI/bge-small-en-v1.5"
    backend: str = "onnx"
    dimension: int = 384
    max_seq_length: int = 512
    batch_size: int = 16


class OCRModelConfig(BaseModel):
    vlm_repo_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-GGUF"
    require_user_preview: bool = True


class ModelsConfig(BaseModel):
    mutual_exclusion_mode: bool = True
    llm: LLMModelConfig = Field(default_factory=LLMModelConfig)
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    ocr: OCRModelConfig = Field(default_factory=OCRModelConfig)


class ChunkingConfig(BaseModel):
    target_tokens: int = 550
    min_tokens: int = 180
    max_tokens: int = 900
    overlap_tokens: int = 60
    preserve_atomic_blocks: bool = True
    route_exercises_to_companion: bool = True


class CompanionConfig(BaseModel):
    keep_multipart_together: bool = True
    quarantine_hints_and_solutions: bool = True
    supported_formats: List[str] = Field(default_factory=lambda: ["pdf", "tex", "md"])


class GraphConfig(BaseModel):
    storage_format: str = "graphar"
    top_k_candidates: int = 12
    similarity_threshold: float = 0.45
    break_cycles: bool = True


class ScoringConfig(BaseModel):
    normalization_epsilon: float = 0.05
    problem_top_k_chunks: int = 5
    problem_similarity_threshold: float = 0.35
    batch_problem_count: int = 4
    dynamic_mastery_update: bool = True


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    offline_katex: bool = True


class GraphyConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    companion: CompanionConfig = Field(default_factory=CompanionConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    def ensure_Runtime_dirs(self, base_dir: Path | None = None) -> None:
        """Ensure all required data and model subdirectories exist."""
        root = base_dir or Path(".")
        data_root = root / self.system.data_dir
        models_root = root / self.system.models_dir
        for sub in (
            data_root / "schemas",
            data_root / "raw",
            data_root / "processed",
            data_root / "vectors",
            data_root / "graphs",
            data_root / "scores",
            data_root / "uploads",
            data_root / "sessions",
            models_root,
        ):
            sub.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path | None = None) -> GraphyConfig:
    """Load and validate GraphyConfig from YAML file or environment default."""
    if config_path is None:
        env_path = os.environ.get("GRAPHY_CONFIG_PATH")
        config_path = Path(env_path) if env_path else Path("configs/config.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return GraphyConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    return GraphyConfig.model_validate(raw_data)
