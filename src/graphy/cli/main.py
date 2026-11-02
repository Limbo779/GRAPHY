"""GRAPHY CLI entrypoint."""

from __future__ import annotations

import json
import click

from graphy.core.config import load_config
from graphy.core.hardware import probe_hardware, render_hardware_table
from graphy.llm.agents import PrerequisiteJudgeAgent, PrerequisitePairInput
from graphy.llm.cache import LLMCache
from graphy.llm.harness import AgentHarness
from graphy.llm.model_manager import ModelLifecycleManager
from graphy.llm.model_setup import setup_models_volume


@click.group(help="GRAPHY & GRAPHY Companion — Self-Discovery Learning System for Math, Physics, and CS.")
def app() -> None:
    """Main CLI command group."""


@app.command("hardware")
@click.option("--json", "as_json", is_flag=True, help="Output hardware profile as JSON.")
def cmd_hardware(as_json: bool) -> None:
    """Probe CPU P/E-cores, RAM budget, and Intel Iris Xe (/dev/dri) GPU access."""
    cfg = load_config()
    profile = probe_hardware(
        dri_node=cfg.hardware.gpu.dri_render_node,
        max_container_cap_gib=cfg.hardware.max_container_ram_gib,
    )
    if as_json:
        click.echo(json.dumps(profile.model_dump(), indent=2))
    else:
        render_hardware_table(profile)


@app.command("config")
@click.option(
    "--path",
    "-c",
    "config_path",
    default="configs/config.yaml",
    help="Path to config.yaml",
)
def cmd_config(config_path: str) -> None:
    """Validate and display the active GRAPHY configuration."""
    cfg = load_config(config_path)
    cfg.ensure_Runtime_dirs()
    click.echo(json.dumps(cfg.model_dump(mode="json"), indent=2))


@app.command("setup-models")
@click.option("--download", is_flag=True, help="Download missing GGUF models from HuggingFace.")
def cmd_setup_models(download: bool) -> None:
    """Configure and verify GGUF models in the persistent ./models/ volume."""
    cfg = load_config()
    manifest = setup_models_volume(
        models_dir=cfg.system.models_dir,
        download_missing=download,
    )
    click.echo(json.dumps(manifest.model_dump(), indent=2))


@app.command("llm-test")
def cmd_llm_test() -> None:
    """Run a live PrerequisiteJudgeAgent verification on Intel Iris Xe Vulkan0."""
    cfg = load_config()
    with ModelLifecycleManager(config=cfg, backend="llama_server") as mgr:
        harness = AgentHarness(model_manager=mgr, cache=LLMCache())
        agent = PrerequisiteJudgeAgent()
        inp = PrerequisitePairInput(
            chunk_a_id="chunk_001",
            chunk_a_breadcrumb="Chapter 1 > 1.1 Vector Spaces",
            chunk_a_text="A vector space V over a field F is a set closed under vector addition and scalar multiplication.",
            chunk_b_id="chunk_014",
            chunk_b_breadcrumb="Chapter 4 > 4.1 Eigenvalues and Eigenvectors",
            chunk_b_text="Let T: V -> V be a linear operator on a vector space V. A scalar lambda is an eigenvalue if T(v) = lambda v.",
        )
        decision = harness.run(agent, inp)
        mem = mgr.get_memory_status()
        click.echo(
            json.dumps(
                {
                    "decision": decision.model_dump(),
                    "memory_status": mem.model_dump(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    app()
