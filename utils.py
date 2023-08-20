from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import os


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second{result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def get_all_paths_in_dir(path, max_paths=None):
    all_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                all_paths.append(os.path.join(root, file))
            if max_paths is not None and len(all_paths) >= max_paths:
                break

    return all_paths


def is_rank_0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
