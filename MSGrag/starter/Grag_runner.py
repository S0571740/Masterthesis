import subprocess
import time
from Hardware_observer import get_system_metrics, get_global_system_metrics


def monitor_system(interval, running_flag, logger):
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


def query_grag(query, type):
    result = subprocess.run(
        [
            "graphrag",
            "query",
            "--root",
            "./MSGrag",
            "--method",
            type,
            "--query",
            query,
            "--community-level",
            "0",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return result


def ms_build_kg():
    subprocess.run(["graphrag", "index", "--root", "./MSGrag"], check=True, text=True)
