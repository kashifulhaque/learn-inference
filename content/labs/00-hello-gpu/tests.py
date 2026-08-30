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
        "ridge point follows from the two peak rates",
        lambda: abs(ridge - 161.2) < 2.0,
        f"got {ridge:.1f} FLOPs/byte from 312 TFLOP/s and 1935 GB/s",
    )

    # Which 80GB A100 you get is up to the provider, and the two variants differ
    # in memory bandwidth. Say so, because it moves every roofline number.
    name = str(report.get("name", ""))
    if "SXM" in name:
        print(
            "\n  This is the SXM4 module, rated at 2039 GB/s rather than the "
            "1935 the\n  starter assumes, so its ridge point is 153, not 161. "
            "Runs land on\n  either variant; chapter 10 covers what that does "
            "to your measurements.",
            flush=True,
        )

    c.metric("gpu_name", report.get("name"))
    c.metric("memory_gb", report.get("memory_gb"))
    c.metric("sm_count", report.get("sm_count"))
    c.metric("capability", report.get("capability"))
    c.metric("ridge_point", round(ridge, 2))
    return c.finish()
