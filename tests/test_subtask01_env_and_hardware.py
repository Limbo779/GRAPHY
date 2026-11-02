"""Subtask 01 verification tests: Config validation, Hardware profiler, and Podman environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner

from graphy.core.config import GraphyConfig, load_config
from graphy.core.hardware import probe_hardware
from graphy.cli.main import app


class TestSubtask01EnvAndHardware(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config("configs/config.yaml")
        self.assertIsInstance(cfg, GraphyConfig)
        self.assertEqual(cfg.system.project_name, "GRAPHY")
        self.assertEqual(cfg.hardware.total_logical_cpus, 12)
        self.assertEqual(cfg.hardware.physical_cores, 10)
        self.assertEqual(cfg.hardware.max_container_ram_gib, 6.5)
        self.assertEqual(cfg.hardware.gpu.dri_render_node, "/dev/dri/renderD128")
        self.assertTrue(cfg.companion.keep_multipart_together)
        self.assertEqual(cfg.scoring.batch_problem_count, 4)
        self.assertTrue(cfg.scoring.dynamic_mastery_update)

    def test_ensure_runtime_dirs(self) -> None:
        cfg = load_config("configs/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg.ensure_Runtime_dirs(base_dir=tmp_path)
            for subdir in (
                "schemas",
                "raw",
                "processed",
                "vectors",
                "graphs",
                "scores",
                "uploads",
                "sessions",
            ):
                self.assertTrue((tmp_path / "data" / subdir).is_dir())
            self.assertTrue((tmp_path / "models").is_dir())

    def test_hardware_probe_i5_1335u_and_iris_xe(self) -> None:
        profile = probe_hardware()
        # Verify CPU topology on the 13th Gen Intel Core i5-1335U
        self.assertEqual(profile.cpu.logical_cpus, 12)
        self.assertEqual(profile.cpu.physical_cores, 10)
        self.assertTrue(profile.cpu.has_avx2)
        self.assertTrue(profile.cpu.has_avx_vnni)
        self.assertEqual(len(profile.cpu.p_core_logical_ids), 4)
        self.assertEqual(len(profile.cpu.e_core_logical_ids), 8)

        # Verify Memory budget detection
        self.assertGreater(profile.memory.total_gib, 14.0)
        self.assertGreater(profile.memory.safe_working_set_gib, 0.5)
        self.assertLessEqual(profile.memory.safe_working_set_gib, 6.5)

        # Verify Intel Iris Xe (/dev/dri/renderD128) access + Vulkan/OpenCL drivers
        self.assertTrue(profile.gpu.node_exists)
        self.assertTrue(profile.gpu.node_accessible)
        self.assertTrue(profile.gpu.is_intel_iris_xe)
        self.assertTrue(profile.gpu.vulkan_available)
        self.assertTrue(profile.gpu.opencl_available)

    def test_cli_hardware_and_config_commands(self) -> None:
        runner = CliRunner()
        res_hw = runner.invoke(app, ["hardware", "--json"])
        self.assertEqual(res_hw.exit_code, 0)
        data = json.loads(res_hw.stdout)
        self.assertTrue(data["gpu"]["is_intel_iris_xe"])

        res_cfg = runner.invoke(app, ["config"])
        self.assertEqual(res_cfg.exit_code, 0)
        cfg_data = json.loads(res_cfg.stdout)
        self.assertEqual(cfg_data["system"]["project_name"], "GRAPHY")


if __name__ == "__main__":
    unittest.main()
