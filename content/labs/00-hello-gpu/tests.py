import torch
from lab_common import Checks


def run(submission):
    c = Checks()
    if not c.require(submission, "gpu_report", "ridge_point"):
        return c.finish()

    c.check("a CUDA device is visible", lambda: torch.cuda.is_available())

    report = submission.gpu_report()
    c.check(
        "gpu_report returns the expected keys",
        lambda: set(report) >= {"name", "memory_gb", "sm_count", "capability"},
        f"got {sorted(report)}",
    )
    c.check(
        "the device has at least 39 GB",
        lambda: report["memory_gb"] >= 39,
        f"{report.get('memory_gb')} GB",
    )
    c.check(
        "the device reports its SM count",
        lambda: report["sm_count"] > 0,
        f"{report.get('sm_count')} SMs",
    )

    ridge = submission.ridge_point()
    c.check(
        "ridge point is about 161 FLOPs/byte for an A100 80GB PCIe",
        lambda: abs(ridge - 161.2) < 2.0,
        f"got {ridge:.1f}",
    )

    c.metric("gpu_name", report.get("name"))
    c.metric("memory_gb", report.get("memory_gb"))
    c.metric("sm_count", report.get("sm_count"))
    c.metric("capability", report.get("capability"))
    c.metric("ridge_point", round(ridge, 2))
    return c.finish()
