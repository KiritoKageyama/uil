"""Machine resource detection — scale UIL workload to ~75% of available capacity."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass


TARGET_FRACTION = 0.75


@dataclass
class ResourceBudget:
    """Suggested parallel/workload knobs derived from this machine."""

    target_fraction: float
    cpu_count_logical: int
    cpu_workers: int
    ram_total_mb: int
    ram_available_mb: int
    ram_budget_mb: int
    gpu_available: bool
    gpu_name: str
    gpu_vram_total_mb: int
    gpu_vram_mb: int
    n_particles: int
    pso_iters: int
    tournament_k: int
    boosting_stages: int
    bagging_estimators: int
    parallel_eval_workers: int
    n_jobs: int
    intra_fit_threads: int
    multi_variants: int
    hybrid_branch_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_info() -> tuple[int, int]:
    """Return (total_mb, available_mb)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except ImportError:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = int(stat.ullTotalPhys // (1024 * 1024))
                avail = int(stat.ullAvailPhys // (1024 * 1024))
                return total, avail
        except Exception:
            pass

    try:
        mem: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, val = line.split(":")
                mem[key.strip()] = int(val.strip().split()[0])
        total = mem.get("MemTotal", 8192 * 1024) // 1024
        avail = mem.get("MemAvailable", mem.get("MemFree", total * 1024)) // 1024
        return int(total), int(avail)
    except Exception:
        return 8192, 4096


def _gpu_vram_mb() -> tuple[bool, int, str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.rsplit(",", 1)]
            if len(parts) == 2:
                name, mb_str = parts[0], parts[1]
                return True, int(float(mb_str)), name
            mb = int(float(line.split(",")[-1].strip()))
            return True, mb, "NVIDIA GPU"
    except Exception:
        pass
    return False, 0, ""


def detect_resource_budget(
    *,
    base_particles: int = 15,
    base_pso_iters: int = 15,
    base_boosting: int = 30,
    target_fraction: float = TARGET_FRACTION,
    telemetry=None,
) -> ResourceBudget:
    """Detect CPU/RAM/GPU and return ~75% utilization workload knobs."""
    frac = float(max(0.25, min(0.95, target_fraction)))
    cpu_logical = os.cpu_count() or 4
    cpu_workers = max(1, int(cpu_logical * frac))

    ram_total, ram_avail = _ram_info()
    ram_budget = max(512, int(min(ram_total, ram_avail) * frac))
    gpu_ok, gpu_total_mb, gpu_name = _gpu_vram_mb()
    gpu_budget_mb = int(gpu_total_mb * frac) if gpu_ok else 0

    # Scale intensity with headroom (4 GB RAM + 4 cores = baseline laptop).
    mem_factor = min(4.0, ram_budget / 3072.0)
    cpu_factor = min(4.0, cpu_workers / 3.0)
    gpu_factor = 1.25 if gpu_ok else 1.0
    scale = max(1.0, min(4.0, mem_factor * cpu_factor * gpu_factor))

    n_particles = int(np_clip_round(
        base_particles * scale, 8, 120, telemetry, "budget_n_particles"))
    pso_iters = int(np_clip_round(
        base_pso_iters * scale, 10, 80, telemetry, "budget_pso_iters"))
    boosting = int(np_clip_round(
        base_boosting * scale, 5, 80, telemetry, "budget_boosting_stages"))
    bagging = int(np_clip_round(
        max(3, base_boosting // 6) * scale, 3, 24, telemetry, "budget_bagging"))
    tournament_k = int(np_clip_round(
        4 + cpu_workers, 4, 16, telemetry, "budget_tournament_k"))
    multi_variants = 5 if cpu_workers >= 6 and ram_budget >= 6144 else 4

    # In-process thread pools (PSO actions, hybrid branches, tournament).
    # Cap at 8: beyond that Python threads oversubscribe BLAS/OpenMP and
    # freeze the whole process (including the web UI).
    parallel_eval = max(1, min(8, cpu_workers - 1 if cpu_logical > 2 else cpu_workers))

    return ResourceBudget(
        target_fraction=frac,
        cpu_count_logical=cpu_logical,
        cpu_workers=cpu_workers,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        ram_budget_mb=ram_budget,
        gpu_available=gpu_ok,
        gpu_name=gpu_name or "",
        gpu_vram_total_mb=gpu_total_mb if gpu_ok else 0,
        gpu_vram_mb=gpu_budget_mb,
        n_particles=n_particles,
        pso_iters=pso_iters,
        tournament_k=tournament_k,
        boosting_stages=boosting,
        bagging_estimators=bagging,
        parallel_eval_workers=parallel_eval,
        n_jobs=cpu_workers,
        intra_fit_threads=max(1, cpu_workers // parallel_eval),
        multi_variants=multi_variants,
        hybrid_branch_count=multi_variants,
    )


def np_clip_round(
    value: float,
    lo: int,
    hi: int,
    telemetry=None,
    name: str | None = None,
) -> int:
    clamped = int(max(lo, min(hi, round(value))))
    if telemetry is not None and name:
        try:
            telemetry.observe(name, float(value), float(clamped))
        except Exception:
            pass
    return clamped

def apply_budget_to_classifier(classifier, budget: ResourceBudget) -> None:
    """Push n_jobs / parallel knobs into sklearn/XGBoost adapters."""
    if hasattr(classifier, "params") and isinstance(classifier.params, dict):
        if "n_jobs" in classifier.params:
            classifier.params["n_jobs"] = budget.n_jobs
    if hasattr(classifier, "apply_config"):
        classifier.apply_config(
            {
                "boosting_stages": budget.boosting_stages,
                "bagging_estimators": budget.bagging_estimators,
                "tournament_k": budget.tournament_k,
                "hybrid_branch_count": budget.hybrid_branch_count,
                "parallel_eval_workers": budget.parallel_eval_workers,
            }
        )
