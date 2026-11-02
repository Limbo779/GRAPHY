"""Hardware profiler and resource optimizer for 13th Gen Intel i5-1335U + Iris Xe."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CPUProfile(BaseModel):
    model_name: str
    logical_cpus: int
    physical_cores: int
    p_core_logical_ids: List[int] = Field(default_factory=list)
    p_core_physical_ids: List[int] = Field(default_factory=list)
    e_core_logical_ids: List[int] = Field(default_factory=list)
    has_avx2: bool = False
    has_avx_vnni: bool = False
    has_fma: bool = False
    recommended_llm_gen_threads: int = 8
    recommended_llm_batch_threads: int = 10
    recommended_physical_cpu_ids: List[int] = Field(default_factory=list)


class MemoryProfile(BaseModel):
    total_gib: float
    available_gib: float
    free_gib: float
    swap_total_gib: float
    cgroup_limit_gib: Optional[float] = None
    safe_working_set_gib: float


class GPUProfile(BaseModel):
    dri_render_node: str = "/dev/dri/renderD128"
    node_exists: bool = False
    node_accessible: bool = False
    driver: Optional[str] = None
    pci_id: Optional[str] = None
    pci_slot: Optional[str] = None
    is_intel_iris_xe: bool = False
    vulkan_available: bool = False
    opencl_available: bool = False
    accelerator_summary: List[str] = Field(default_factory=list)


class HardwareProfile(BaseModel):
    in_container: bool
    container_engine: Optional[str] = None
    cpu: CPUProfile
    memory: MemoryProfile
    gpu: GPUProfile


def _detect_container() -> tuple[bool, Optional[str]]:
    if Path("/run/.containerenv").exists():
        return True, "podman"
    if Path("/.dockerenv").exists():
        return True, "docker"
    env_container = os.environ.get("container")
    if env_container:
        return True, env_container
    return False, None


def _probe_cpu() -> CPUProfile:
    model_name = "Unknown CPU"
    flags: set[str] = set()
    logical_cpus = os.cpu_count() or 1

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name") and model_name == "Unknown CPU":
                _, _, val = line.partition(":")
                model_name = val.strip()
            elif line.startswith("flags") and not flags:
                _, _, val = line.partition(":")
                flags = set(val.strip().split())

    # Detect hybrid P-cores vs E-cores via sysfs max frequency & thread siblings
    sys_cpu_root = Path("/sys/devices/system/cpu")
    max_freqs: Dict[int, int] = {}
    first_sibling_of: Dict[int, int] = {}

    for cpu_id in range(logical_cpus):
        cpu_dir = sys_cpu_root / f"cpu{cpu_id}"
        freq_file = cpu_dir / "cpufreq" / "cpuinfo_max_freq"
        if freq_file.exists():
            try:
                max_freqs[cpu_id] = int(freq_file.read_text().strip())
            except ValueError:
                pass

        sib_file = cpu_dir / "topology" / "thread_siblings_list"
        if sib_file.exists():
            raw = sib_file.read_text().strip()
            parts = []
            for token in raw.split(","):
                if "-" in token:
                    start, end = token.split("-", 1)
                    parts.extend(range(int(start), int(end) + 1))
                elif token.isdigit():
                    parts.append(int(token))
            if parts:
                first_sibling_of[cpu_id] = min(parts)
        else:
            first_sibling_of[cpu_id] = cpu_id

    p_core_logical: List[int] = []
    e_core_logical: List[int] = []

    if max_freqs and len(set(max_freqs.values())) > 1:
        top_freq = max(max_freqs.values())
        for cpu_id in range(logical_cpus):
            if max_freqs.get(cpu_id, 0) == top_freq:
                p_core_logical.append(cpu_id)
            else:
                e_core_logical.append(cpu_id)
    else:
        p_core_logical = list(range(logical_cpus))

    physical_rep_ids = sorted(set(first_sibling_of.values()))
    physical_cores = len(physical_rep_ids) if physical_rep_ids else logical_cpus
    p_core_physical = [c for c in physical_rep_ids if c in p_core_logical]

    gen_threads = min(8, physical_cores)
    batch_threads = min(10, logical_cpus)

    return CPUProfile(
        model_name=model_name,
        logical_cpus=logical_cpus,
        physical_cores=physical_cores,
        p_core_logical_ids=p_core_logical,
        p_core_physical_ids=p_core_physical,
        e_core_logical_ids=e_core_logical,
        has_avx2="avx2" in flags,
        has_avx_vnni="avx_vnni" in flags,
        has_fma="fma" in flags,
        recommended_llm_gen_threads=gen_threads,
        recommended_llm_batch_threads=batch_threads,
        recommended_physical_cpu_ids=physical_rep_ids,
    )


def _probe_memory(max_container_cap_gib: float = 6.5) -> MemoryProfile:
    mem_kb: Dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            val_str = rest.strip().split()[0] if rest.strip() else "0"
            if val_str.isdigit():
                mem_kb[key] = int(val_str)

    kb_to_gib = 1.0 / (1024.0 * 1024.0)
    total_gib = round(mem_kb.get("MemTotal", 0) * kb_to_gib, 2)
    available_gib = round(mem_kb.get("MemAvailable", 0) * kb_to_gib, 2)
    free_gib = round(mem_kb.get("MemFree", 0) * kb_to_gib, 2)
    swap_total_gib = round(mem_kb.get("SwapTotal", 0) * kb_to_gib, 2)

    cgroup_limit_gib: Optional[float] = None
    cgroup_max_path = Path("/sys/fs/cgroup/memory.max")
    if cgroup_max_path.exists():
        raw_max = cgroup_max_path.read_text().strip()
        if raw_max.isdigit():
            cgroup_limit_gib = round(int(raw_max) / (1024.0**3), 2)

    effective_avail = available_gib
    if cgroup_limit_gib is not None:
        effective_avail = min(effective_avail, cgroup_limit_gib)

    safe_working_set = round(min(max_container_cap_gib, effective_avail * 0.75), 2)

    return MemoryProfile(
        total_gib=total_gib,
        available_gib=available_gib,
        free_gib=free_gib,
        swap_total_gib=swap_total_gib,
        cgroup_limit_gib=cgroup_limit_gib,
        safe_working_set_gib=safe_working_set,
    )


def _probe_gpu(dri_node: str = "/dev/dri/renderD128") -> GPUProfile:
    node_path = Path(dri_node)
    exists = node_path.exists()
    accessible = os.access(dri_node, os.R_OK | os.W_OK) if exists else False

    driver: Optional[str] = None
    pci_id: Optional[str] = None
    pci_slot: Optional[str] = None

    uevent_path = Path(f"/sys/class/drm/{node_path.name}/device/uevent")
    if uevent_path.exists():
        for line in uevent_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("DRIVER="):
                driver = line.split("=", 1)[1].strip()
            elif line.startswith("PCI_ID="):
                pci_id = line.split("=", 1)[1].strip()
            elif line.startswith("PCI_SLOT_NAME="):
                pci_slot = line.split("=", 1)[1].strip()

    is_intel = bool(pci_id and pci_id.upper().startswith("8086:")) or driver in ("i915", "xe")

    accelerators: List[str] = ["CPU-AVX2-VNNI"]
    vulkan_ok = False
    opencl_ok = False

    if accessible and shutil.which("clinfo"):
        try:
            proc = subprocess.run(
                ["clinfo", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "Intel" in proc.stdout or "Iris" in proc.stdout or "Device #0" in proc.stdout:
                opencl_ok = True
                accelerators.append("OpenCL-Intel-Iris-Xe")
        except Exception:
            pass

    if accessible and shutil.which("vulkaninfo"):
        try:
            proc = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if "Intel" in proc.stdout or "Iris" in proc.stdout or "GPU0" in proc.stdout:
                vulkan_ok = True
                accelerators.append("Vulkan-Intel-Iris-Xe")
        except Exception:
            pass

    return GPUProfile(
        dri_render_node=dri_node,
        node_exists=exists,
        node_accessible=accessible,
        driver=driver,
        pci_id=pci_id,
        pci_slot=pci_slot,
        is_intel_iris_xe=is_intel,
        vulkan_available=vulkan_ok,
        opencl_available=opencl_ok,
        accelerator_summary=accelerators,
    )


def probe_hardware(
    dri_node: str = "/dev/dri/renderD128",
    max_container_cap_gib: float = 6.5,
) -> HardwareProfile:
    """Probe CPU topology, RAM budget, and Intel Iris Xe /dev/dri access."""
    in_container, engine = _detect_container()
    return HardwareProfile(
        in_container=in_container,
        container_engine=engine,
        cpu=_probe_cpu(),
        memory=_probe_memory(max_container_cap_gib=max_container_cap_gib),
        gpu=_probe_gpu(dri_node=dri_node),
    )


def render_hardware_table(profile: HardwareProfile) -> str:
    """Format a clean diagnostic table for the hardware profile."""
    rows = [
        (
            "Runtime",
            "Container Environment",
            f"Yes ({profile.container_engine})" if profile.in_container else "Host OS (Arch Linux)",
        ),
        ("CPU", "Model", profile.cpu.model_name),
        (
            "CPU",
            "Topology (Hybrid P/E)",
            f"{profile.cpu.logical_cpus} logical / {profile.cpu.physical_cores} physical "
            f"(P={profile.cpu.p_core_logical_ids}, E={profile.cpu.e_core_logical_ids})",
        ),
        (
            "CPU",
            "SIMD Acceleration",
            f"AVX2={profile.cpu.has_avx2}, AVX-VNNI={profile.cpu.has_avx_vnni}, FMA={profile.cpu.has_fma}",
        ),
        (
            "CPU",
            "Recommended Threads",
            f"LLM Gen={profile.cpu.recommended_llm_gen_threads}, Batch={profile.cpu.recommended_llm_batch_threads}",
        ),
        (
            "Memory",
            "System RAM (GiB)",
            f"Total={profile.memory.total_gib} GiB, Available={profile.memory.available_gib} GiB",
        ),
        (
            "Memory",
            "Safe Container Budget",
            f"{profile.memory.safe_working_set_gib} GiB (Cgroup Limit={profile.memory.cgroup_limit_gib or 'unlimited'})",
        ),
        (
            "GPU (iGPU)",
            "DRI Render Node",
            f"{profile.gpu.dri_render_node} (exists={profile.gpu.node_exists}, rw_access={profile.gpu.node_accessible})",
        ),
        (
            "GPU (iGPU)",
            "Intel Iris Xe Driver / PCI",
            f"driver={profile.gpu.driver}, pci_id={profile.gpu.pci_id}, intel_xe={profile.gpu.is_intel_iris_xe}",
        ),
        (
            "GPU (iGPU)",
            "Compute Backends",
            f"Vulkan={profile.gpu.vulkan_available}, OpenCL={profile.gpu.opencl_available} "
            f"({', '.join(profile.gpu.accelerator_summary)})",
        ),
    ]

    lines = [
        "=" * 92,
        " GRAPHY Hardware & Rootless Podman Container Profile",
        "=" * 92,
        f"{'Subsystem':<14} | {'Property':<26} | {'Detected Value'}",
        "-" * 92,
    ]
    for sub, prop, val in rows:
        lines.append(f"{sub:<14} | {prop:<26} | {val}")
    lines.append("=" * 92)
    output = "\n".join(lines)
    print(output)
    return output


if __name__ == "__main__":
    prof = probe_hardware()
    render_hardware_table(prof)
