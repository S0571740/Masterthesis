import struct
import time
import os
from Hardware_observer import get_system_metrics, get_global_system_metrics

# Formats
RECORD_FORMAT_GLOBAL = "fff"  # total_ram_gb, disk_total_gb, disk_start_free_gb
RECORD_FORMAT_INTERVAL = (
    "Ifff"  # seconds_offset, cpu_percent, ram_available_gb, disk_free_gb
)

RECORD_SIZE_GLOBAL = struct.calcsize(RECORD_FORMAT_GLOBAL)
RECORD_SIZE_INTERVAL = struct.calcsize(RECORD_FORMAT_INTERVAL)


class Logger:
    def __init__(self, file="metrics/metrics"):
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
        self.global_file = f"{file}_GLOBAL.bin"
        self.interval_file = f"{file}_INTERVAL.bin"
        self.start_time = time.time()

    def log_global(self, total_ram_gb, disk_total_gb, disk_start_free_gb):
        binary_record = struct.pack(
            RECORD_FORMAT_GLOBAL, total_ram_gb, disk_total_gb, disk_start_free_gb
        )
        with open(self.global_file, "wb") as f:
            f.write(binary_record)

    def log(self, cpu_percent, ram_available_gb, disk_free_gb):
        elapsed_seconds = int(time.time() - self.start_time)
        binary_record = struct.pack(
            RECORD_FORMAT_INTERVAL,
            elapsed_seconds,
            cpu_percent,
            ram_available_gb,
            disk_free_gb,
        )
        with open(self.interval_file, "ab") as f:
            f.write(binary_record)

    def get_interval_log_file(self):
        return self.interval_file

    def get_global_log_file(self):
        return self.global_file
    
    def get_root_file_path(self):
        return os.path.abspath(os.path.dirname(self.global_file))

def monitor_system(logger, interval, running_flag):
    global_metrics = get_global_system_metrics()
    logger.log_global(
        total_ram_gb=global_metrics["s"],
        disk_total_gb=global_metrics["e"],
        disk_start_free_gb=global_metrics["f"],
    )

    while running_flag["running"]:
        metrics = get_system_metrics()
        logger.log(
            cpu_percent=metrics.get("c", 0.0),
            ram_available_gb=metrics["r"],
            disk_free_gb=metrics["d"],
        )
        time.sleep(interval)
