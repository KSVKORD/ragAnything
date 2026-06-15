"""Performance recording: time + CPU/RAM/GPU sampled per named process stage,
written to a markdown + JSON report document."""
import os
import json
import time
import platform
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
    _GPU = torch.cuda.is_available()
except Exception:
    torch = None
    _GPU = False

REPORT_DIR = os.getenv("METRICS_DIR") or os.path.join(os.getenv("OUTPUT_DIR", "./output"), "reports")
SAMPLE_INTERVAL = float(os.getenv("METRICS_INTERVAL", "0.5"))

_records = []


class _Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.cpu, self.gpu_util = [], []
        self.ram_peak = self.gpu_mem_peak = 0.0

    def run(self):
        if psutil:
            psutil.cpu_percent(None)  # prime; first call returns 0.0
        while not self._stop.is_set():
            if psutil:
                self.cpu.append(psutil.cpu_percent(None))
                self.ram_peak = max(self.ram_peak, psutil.virtual_memory().used / 1e9)
            if _GPU:
                try:
                    self.gpu_mem_peak = max(self.gpu_mem_peak, torch.cuda.memory_reserved() / 1e6)
                except Exception:
                    pass
                try:
                    self.gpu_util.append(torch.cuda.utilization())  # needs pynvml; ignored if missing
                except Exception:
                    pass
            self._stop.wait(SAMPLE_INTERVAL)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


@contextmanager
def record(name):
    """Time a named stage and sample resources while it runs; yields a dict whose
    'extra' can be filled with stage-specific counts."""
    rec = {"name": name, "start": datetime.now().isoformat(timespec="seconds"), "extra": {}}
    sampler = _Sampler()
    if _GPU:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    t0 = time.perf_counter()
    sampler.start()
    try:
        yield rec
    finally:
        rec["duration_s"] = round(time.perf_counter() - t0, 2)
        sampler.stop()
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
        rec["cpu_avg"] = avg(sampler.cpu)
        rec["cpu_peak"] = round(max(sampler.cpu), 1) if sampler.cpu else None
        rec["ram_peak_gb"] = round(sampler.ram_peak, 2) if sampler.ram_peak else None
        rec["gpu_mem_peak_mb"] = round(sampler.gpu_mem_peak) if sampler.gpu_mem_peak else None
        rec["gpu_util_avg"] = avg(sampler.gpu_util)
        rec["gpu_util_peak"] = max(sampler.gpu_util) if sampler.gpu_util else None
        _records.append(rec)


def _environment():
    info = [f"- Host: {platform.node()} ({platform.system()} {platform.release()})",
            f"- Python: {platform.python_version()}"]
    if psutil:
        info.append(f"- CPU cores: {psutil.cpu_count(logical=True)}")
        info.append(f"- Total RAM: {round(psutil.virtual_memory().total / 1e9, 1)} GB")
    if _GPU:
        try:
            info.append(f"- GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        except Exception:
            info.append("- GPU: available")
    else:
        info.append("- GPU: not available (CPU mode)")
    return info


def write_report(title="process", out_dir=None, clear=True):
    """Write the collected stages to <out_dir>/perf_<title>_<ts>.md (+ .json). Returns the .md path."""
    if not _records:
        return None
    out = Path(out_dir or REPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out / f"perf_{title}_{ts}"

    total = round(sum(r.get("duration_s") or 0 for r in _records), 2)
    lines = [f"# Performance report — {title}", "",
             f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
             "## Environment", "", *_environment(), "",
             "## Stages", "",
             "| Process | Start | Duration (s) | CPU avg/peak % | RAM peak (GB) | "
             "GPU mem peak (MB) | GPU util avg/peak % | Notes |",
             "|---|---|---|---|---|---|---|---|"]
    for r in _records:
        notes = ", ".join(f"{k}={v}" for k, v in r["extra"].items())
        lines.append(
            f"| {r['name']} | {r['start']} | {r.get('duration_s')} | "
            f"{r.get('cpu_avg')}/{r.get('cpu_peak')} | {r.get('ram_peak_gb')} | "
            f"{r.get('gpu_mem_peak_mb')} | {r.get('gpu_util_avg')}/{r.get('gpu_util_peak')} | {notes} |")
    lines += ["", f"**Total wall time: {total} s across {len(_records)} stage(s).**", ""]

    base.with_suffix(".md").write_text("\n".join(lines))
    base.with_suffix(".json").write_text(json.dumps(
        {"title": title, "generated": ts, "environment": _environment(), "stages": _records}, indent=2))
    if clear:
        _records.clear()
    return str(base.with_suffix(".md"))
