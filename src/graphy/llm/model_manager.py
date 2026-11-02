"""Memory-Strict Local AI Model Lifecycle Manager for i5-1335U + 15 GiB RAM + Intel Iris Xe."""

from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
import psutil
from pydantic import BaseModel

from graphy.core.config import GraphyConfig, load_config
from graphy.core.hardware import probe_hardware
from graphy.core.logging import setup_logging

logger = setup_logging()

ModelRole = Literal["instruct", "embedding", "vision", "reasoning"]


class MemoryStatus(BaseModel):
    active_role: Optional[str] = None
    active_pid: Optional[int] = None
    server_rss_gib: float = 0.0
    container_rss_gib: float = 0.0
    system_available_gib: float = 0.0
    max_budget_gib: float = 6.5
    within_budget: bool = True


class ModelLifecycleManager:
    """Manages `llama-server` lifecycle with strict mutual exclusion and < 6.0 GiB RAM cap."""

    def __init__(
        self,
        config: Optional[GraphyConfig] = None,
        backend: Literal["llama_server", "mock"] = "llama_server",
        mock_handler: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config or load_config()
        self.backend = backend
        self.mock_handler = mock_handler
        self._proc: Optional[subprocess.Popen[str]] = None
        self._active_role: Optional[ModelRole] = None
        self._active_model_id: Optional[str] = None

    @property
    def active_role(self) -> Optional[ModelRole]:
        return self._active_role

    @property
    def active_model_id(self) -> str:
        return self._active_model_id or self.model_id_for_role(self._active_role or "instruct")

    def model_id_for_role(self, role: ModelRole) -> str:
        if self.backend == "mock":
            return f"mock-{role}"
        if role == "instruct":
            return self.config.models.llm.filename
        if role == "embedding":
            return f"{self.config.models.embedding.filename}:embedding"
        if role == "vision":
            return f"{self.config.models.ocr.llm_filename}+{self.config.models.ocr.mmproj_filename}"
        if role == "reasoning":
            return self.config.models.llm.reasoning_filename
        return self.config.models.llm.filename

    @property
    def base_url(self) -> str:
        return f"http://{self.config.models.server_host}:{self.config.models.server_port}"

    def _resolve_server_bin(self) -> Path:
        candidates = [
            Path(self.config.models.llama_server_bin),
            Path("/opt/llama/llama-server"),
            Path("/tmp/opencode/ai_runtime/lib/ollama/llama-server"),
        ]
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return c
        raise FileNotFoundError(
            f"llama-server binary not found in {[str(c) for c in candidates]}"
        )

    def _resolve_lib_env(self, server_bin: Path) -> Dict[str, str]:
        env = dict(os.environ)
        lib_dirs = [
            str(server_bin.parent),
            str(server_bin.parent / "vulkan"),
            self.config.models.llama_lib_dir,
        ]
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            lib_dirs.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
        return env

    def _resolve_model_path(self, filename: str) -> Path:
        models_dir = Path(self.config.system.models_dir)
        candidate = models_dir / filename
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Model file '{filename}' not found in '{models_dir}'. Run `graphy setup-models` first."
        )

    def get_memory_status(self) -> MemoryStatus:
        hw = probe_hardware(
            dri_node=self.config.hardware.gpu.dri_render_node,
            max_container_cap_gib=self.config.hardware.max_container_ram_gib,
        )
        server_rss = 0.0
        pid: Optional[int] = None
        if self._proc is not None and self._proc.poll() is None:
            pid = self._proc.pid
            try:
                p = psutil.Process(pid)
                server_rss = round(p.memory_info().rss / (1024.0**3), 3)
            except psutil.Error:
                server_rss = 0.0

        self_rss = round(psutil.Process(os.getpid()).memory_info().rss / (1024.0**3), 3)
        total_rss = round(server_rss + self_rss, 3)
        max_budget = self.config.hardware.max_container_ram_gib

        return MemoryStatus(
            active_role=self._active_role,
            active_pid=pid,
            server_rss_gib=server_rss,
            container_rss_gib=total_rss,
            system_available_gib=hw.memory.available_gib,
            max_budget_gib=max_budget,
            within_budget=(total_rss <= max_budget),
        )

    def release_all(self) -> None:
        """Stop any running llama-server subprocess and reclaim system/iGPU RAM."""
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=4)
            if self._proc.stdout is not None:
                try:
                    self._proc.stdout.close()
                except Exception:
                    pass
            self._proc = None
        self._active_role = None
        self._active_model_id = None

    def acquire_role(self, role: ModelRole) -> None:
        """Ensure the requested model role is loaded, unloading any conflicting role first."""
        if self.backend == "mock":
            if self.config.models.mutual_exclusion_mode and self._active_role != role:
                self.release_all()
            self._active_role = role
            self._active_model_id = f"mock-{role}"
            return

        if (
            self._proc is not None
            and self._proc.poll() is None
            and self._active_role == role
            and self._is_server_healthy()
        ):
            return

        # Mutual exclusion: unload previous role before loading new role
        self.release_all()

        mem = self.get_memory_status()
        if mem.system_available_gib < self.config.hardware.min_free_ram_for_llm_gib:
            logger.warning(
                f"Low available RAM ({mem.system_available_gib} GiB). Reducing GPU layers & context window."
            )
            low_mem_mode = True
        else:
            low_mem_mode = False

        server_bin = self._resolve_server_bin()
        env = self._resolve_lib_env(server_bin)

        hw = probe_hardware(
            dri_node=self.config.hardware.gpu.dri_render_node,
            max_container_cap_gib=self.config.hardware.max_container_ram_gib,
        )
        use_vulkan = hw.gpu.vulkan_available and self.config.hardware.gpu.enabled
        device_arg = self.config.hardware.gpu.vulkan_device if use_vulkan else "none"

        gen_threads = str(self.config.hardware.llm_generation_threads)
        batch_threads = str(self.config.hardware.llm_batch_threads)

        cmd: List[str] = [
            str(server_bin),
            "--host",
            self.config.models.server_host,
            "--port",
            str(self.config.models.server_port),
            "--device",
            device_arg,
            "-t",
            gen_threads,
            "-tb",
            batch_threads,
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
        ]

        if role == "instruct":
            model_file = self._resolve_model_path(self.config.models.llm.filename)
            ngl = 8 if low_mem_mode else self.config.models.llm.gpu_layers
            ctx = 2048 if low_mem_mode else self.config.models.llm.context_window
            cmd.extend(["-m", str(model_file), "-ngl", str(ngl if use_vulkan else 0), "-c", str(ctx)])
            self._active_model_id = self.config.models.llm.filename

        elif role == "embedding":
            model_file = self._resolve_model_path(self.config.models.embedding.filename)
            ngl = 20 if low_mem_mode else self.config.models.embedding.gpu_layers
            ctx = self.config.models.embedding.max_seq_length
            cmd.extend(
                [
                    "-m",
                    str(model_file),
                    "-ngl",
                    str(ngl if use_vulkan else 0),
                    "-c",
                    str(ctx),
                    "--embedding",
                    "--pooling",
                    "mean",
                ]
            )
            self._active_model_id = f"{self.config.models.embedding.filename}:embedding"

        elif role == "vision":
            model_file = self._resolve_model_path(self.config.models.ocr.llm_filename)
            mmproj_file = self._resolve_model_path(self.config.models.ocr.mmproj_filename)
            ngl = 8 if low_mem_mode else self.config.models.ocr.gpu_layers
            cmd.extend(
                [
                    "-m",
                    str(model_file),
                    "--mmproj",
                    str(mmproj_file),
                    "-ngl",
                    str(ngl if use_vulkan else 0),
                    "-c",
                    "2048",
                ]
            )
            self._active_model_id = (
                f"{self.config.models.ocr.llm_filename}+{self.config.models.ocr.mmproj_filename}"
            )

        elif role == "reasoning":
            model_file = self._resolve_model_path(self.config.models.llm.reasoning_filename)
            ngl = 8 if low_mem_mode else 16
            cmd.extend(["-m", str(model_file), "-ngl", str(ngl if use_vulkan else 0), "-c", "4096"])
            self._active_model_id = self.config.models.llm.reasoning_filename

        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._active_role = role

        # Wait for health check
        deadline = time.time() + 45.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = self._proc.stdout.read() if self._proc.stdout else ""
                self.release_all()
                raise RuntimeError(f"llama-server exited prematurely while loading role '{role}': {out}")
            if self._is_server_healthy():
                return
            time.sleep(0.35)

        self.release_all()
        raise TimeoutError(f"Timed out waiting for llama-server role '{role}' to become healthy.")

    def _is_server_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.0) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as err:
            err.close()
            return False
        except Exception:
            return False

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str = "AgentOutput",
        role: ModelRole = "instruct",
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        image_path: Optional[Path] = None,
    ) -> str:
        """Execute a grammar-constrained JSON-schema chat completion."""
        target_role: ModelRole = "vision" if image_path is not None else role
        self.acquire_role(target_role)

        if self.backend == "mock":
            if self.mock_handler is not None:
                res_dict = self.mock_handler(
                    schema_name,
                    {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "schema": json_schema,
                        "image_path": str(image_path) if image_path else None,
                    },
                )
                return json.dumps(res_dict)
            raise RuntimeError("Mock backend requires a mock_handler callable.")

        user_content: Any
        if image_path is not None:
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            b64_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
                {"type": "text", "text": user_prompt},
            ]
        else:
            user_content = user_prompt

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature if temperature is not None else self.config.models.llm.temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        return str(body["choices"][0]["message"]["content"])

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute L2-normalized dense embeddings for a batch of texts."""
        if not texts:
            return []

        self.acquire_role("embedding")

        if self.backend == "mock":
            # Deterministic unit-length pseudo-embeddings for testing
            dim = 64
            results: List[List[float]] = []
            for t in texts:
                vec = [math.sin((i + 1) * (ord(c) if c else 1)) for i, c in enumerate((t * dim)[:dim])]
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                results.append([round(v / norm, 6) for v in vec])
            return results

        batch_size = self.config.models.embedding.batch_size
        all_vectors: List[List[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"input": batch}
            req = urllib.request.Request(
                f"{self.base_url}/v1/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            for item in body["data"]:
                raw_vec: List[float] = item["embedding"]
                norm = math.sqrt(sum(x * x for x in raw_vec)) or 1.0
                all_vectors.append([x / norm for x in raw_vec])

        return all_vectors

    def __enter__(self) -> "ModelLifecycleManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_all()
