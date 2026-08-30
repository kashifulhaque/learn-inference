import torch

PEAK_BF16_TFLOPS = 312.0
PEAK_BANDWIDTH_GBS = 1935.0


def gpu_report() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "memory_gb": round(props.total_memory / 1024**3, 1),
        "sm_count": props.multi_processor_count,
        "capability": f"{props.major}.{props.minor}",
    }


def ridge_point(tflops: float = PEAK_BF16_TFLOPS,
                bandwidth_gbs: float = PEAK_BANDWIDTH_GBS) -> float:
    return (tflops * 1e12) / (bandwidth_gbs * 1e9)
