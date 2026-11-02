"""GRAPHY CLI entrypoint."""

from __future__ import annotations

import json
import click

from graphy.core.config import load_config
from graphy.core.hardware import probe_hardware, render_hardware_table


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


if __name__ == "__main__":
    app()
