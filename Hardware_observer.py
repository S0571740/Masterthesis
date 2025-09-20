import psutil
import GPUtil
from datetime import datetime

# t = timestamp
# s = total ram
# e = total disk
# f = free disk space at start

# n = gpu name
# l = gpu load percentage
# m = gpu memory used percentage
# n = gpu memory total

# r = available ram
# d = free disk space
# c = cpu percentage

def get_global_system_metrics():
    metrics = {
        "t": datetime.now().isoformat(),
        "s": psutil.virtual_memory().total / (1024 ** 3),
        "e": psutil.disk_usage('/').total / (1024 ** 3),
        "f": psutil.disk_usage('/').free / (1024 ** 3),
    }
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metrics.update({
                "n": gpu.name,
                "l": gpu.load * 100,
                "m": gpu.memoryUtil * 100,
                "n": gpu.memoryTotal / 1024,
            })
    except Exception as e:
        metrics["gpu_error"] = str(e)
    return metrics

def get_system_metrics():
    metrics = {
        "c": psutil.cpu_percent(interval=None),
        "r": psutil.virtual_memory().available / (1024 ** 3),
        "d": psutil.disk_usage('/').free / (1024 ** 3),
    }
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metrics.update({
                "l": gpu.load * 100,
                "m": gpu.memoryUtil * 100,
            })
    except Exception as e:
        metrics["gpu_error"] = str(e)
    return metrics