import struct
import csv
import json
import os
import statistics

# Global file format
RECORD_FORMAT_GLOBAL = "fff"
RECORD_SIZE_GLOBAL = struct.calcsize(RECORD_FORMAT_GLOBAL)

# Interval file format
RECORD_FORMAT_INTERVAL = "Ifff"
RECORD_SIZE_INTERVAL = struct.calcsize(RECORD_FORMAT_INTERVAL)


def convert_bin_to_file(file_path, mode="csv", preview=False):
    convert_global_bin(file_path + "_GLOBAL.bin", mode)
    convert_interval_bin(file_path + "_INTERVAL.bin", preview, mode)


def final_merge(input_dir, mode="csv"):

    data = {
        "ms_global": {"ram": [], "disk_total": [], "disk_free": []},
        "neo_global": {"ram": [], "disk_total": [], "disk_free": []},
        "ms_interval": {"seconds": [], "cpu": [], "ram": [], "disk": []},
        "neo_interval": {"seconds": [], "cpu": [], "ram": [], "disk": []},
    }

    for ms_neo_dirs in os.listdir(input_dir):
        ms_neo_dirs = os.path.join(input_dir, ms_neo_dirs)
        if os.path.isdir(ms_neo_dirs):
            for merged_file in os.listdir(ms_neo_dirs):
                merged_file = os.path.join(ms_neo_dirs, merged_file)
                if not os.path.isdir(merged_file):
                    with open(merged_file, "rb") as f:
                        if "GLOBAL" in merged_file:
                            ram, disk_total, disk_free = struct.unpack(
                                RECORD_FORMAT_GLOBAL, f.read(RECORD_SIZE_GLOBAL)
                            )
                            target = (
                                "ms_global" if "ms" in ms_neo_dirs else "neo_global"
                            )
                            data[target]["ram"].append(ram)
                            data[target]["disk_total"].append(disk_total)
                            data[target]["disk_free"].append(disk_free)
                        elif "INTERVAL" in merged_file:
                            target = (
                                "ms_interval" if "ms" in ms_neo_dirs else "neo_interval"
                            )
                            while True:
                                record = f.read(RECORD_SIZE_INTERVAL)
                                if not record:
                                    break
                                seconds, cpu, ram, disk = struct.unpack(
                                    RECORD_FORMAT_INTERVAL, record
                                )
                                data[target]["seconds"].append(seconds)
                                data[target]["cpu"].append(cpu)
                                data[target]["ram"].append(ram)
                                data[target]["disk"].append(disk)
                        else:
                            print(
                                f"File could not been identified, path was:{merged_file}."
                            )
    ms_global_merged = merge_global(**data["ms_global"])
    neo_global_merged = merge_global(**data["neo_global"])
    ms_interval_merged = merge_interval(**data["ms_interval"])
    neo_interval_merged = merge_interval(**data["neo_interval"])

    ms_file = f"{input_dir}/final_merge_ms"
    neo_file = f"{input_dir}/final_merge_neo"

    write_global(f"{ms_file}_GLOBAL.{mode}", ms_global_merged, mode)
    write_interval(f"{ms_file}_INTERVAL.{mode}", ms_interval_merged, mode)

    write_global(f"{neo_file}_GLOBAL.{mode}", neo_global_merged, mode)
    write_interval(f"{neo_file}_INTERVAL.{mode}", neo_interval_merged, mode)


def merge_global(ram, disk_total, disk_free):
    return [
        {"total_ram_gb": r, "disk_total_gb": dt, "disk_start_free_gb": df}
        for r, dt, df in zip(ram, disk_total, disk_free)
    ]


def merge_interval(seconds, cpu, ram, disk):
    return [
        {"seconds": s, "cpu_percent": c, "ram_available_gb": r, "disk_free_gb": d}
        for s, c, r, d in zip(seconds, cpu, ram, disk)
    ]


def compute_stats(values, source=""):
    return {
        f"{source}_mean": statistics.mean(values),
        f"{source}_median": statistics.median(values),
        f"{source}_stdev": statistics.stdev(values),
        f"{source}_variance": statistics.variance(values),
        f"{source}_min": min(values),
        f"{source}_max": max(values),
    }


def print_stats(name, stats):
    print(
        f"{name} — Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, "
        f"Std Dev: {stats['stdev']:.2f}, Variance: {stats['variance']:.2f}, "
        f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}"
    )


def merge(input_dir, output_file, mode="csv"):
    global_ram = []
    global_disk_total = []
    global_disk_free = []

    interval_seconds = []
    interval_cpu_percent = []
    interval_ram_available = []
    interval_disk_free = []
    for file in os.listdir(input_dir):
        if not os.path.isdir(file):
            _, extension = os.path.splitext(os.path.join(input_dir, file))
            if extension == ".bin":
                with open(os.path.join(input_dir, file), "rb") as f:
                    if "GLOBAL" in file:
                        record = f.read(RECORD_SIZE_GLOBAL)
                        if not record:
                            print("Global file is empty.")
                            return
                        ram, disk_total, disk_free = struct.unpack(
                            RECORD_FORMAT_GLOBAL, record
                        )
                        global_ram.append(ram)
                        global_disk_total.append(disk_total)
                        global_disk_free.append(disk_free)
                    elif "INTERVAL" in file:
                        while True:
                            record = f.read(RECORD_SIZE_INTERVAL)
                            if not record:
                                break
                            seconds, cpu, ram, disk = struct.unpack(
                                RECORD_FORMAT_INTERVAL, record
                            )
                            interval_seconds.append(seconds)
                            interval_cpu_percent.append(cpu)
                            interval_ram_available.append(ram)
                            interval_disk_free.append(disk)
                    else:
                        print(f"File could not been identified, path was:{file}.")
    global_merged = [
        {"total_ram_gb": x, "disk_total_gb": y, "disk_start_free_gb": z}
        for x, y, z in zip(global_ram, global_disk_total, global_disk_free)
    ]
    interval_merged = [
        {"seconds": x, "cpu_percent": y, "ram_available_gb": z, "disk_free_gb": w}
        for x, y, z, w in zip(
            interval_seconds,
            interval_cpu_percent,
            interval_ram_available,
            interval_disk_free,
        )
    ]
    write_global(f"{output_file}_GLOBAL.{mode}", global_merged, mode)
    write_interval(f"{output_file}_INTERVAL.{mode}", interval_merged, mode)


def write_global(output_file, records, mode="csv"):
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if isinstance(records, dict):
        records = [records]
    if mode == "json":
        with open(output_file, "w") as jsonfile:
            json.dump(records, jsonfile, indent=4)
    elif mode == "csv":
        with open(output_file, "w", newline="") as csvfile:
            fieldnames = ["total_ram_gb", "disk_total_gb", "disk_start_free_gb"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for records_row in records:
                writer.writerow(records_row)
    elif mode == "bin":
        with open(output_file, "wb") as binfile:
            for record in records:
                packed = struct.pack(
                    RECORD_FORMAT_GLOBAL,
                    record["total_ram_gb"],
                    record["disk_total_gb"],
                    record["disk_start_free_gb"],
                )
                binfile.write(packed)
    else:
        print("Unsupported mode. Please select 'csv' or 'json'.")
        return
    print(f"Global metrics written to {output_file}")


def write_interval(output_file, records, mode="csv"):
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if isinstance(records, dict):
        records = [records]
    if mode == "json":
        with open(output_file, "w") as jsonfile:
            json.dump(records, jsonfile, indent=4)
    elif mode == "csv":
        with open(output_file, "w", newline="") as csvfile:
            fieldnames = [
                "seconds",
                "cpu_percent",
                "ram_available_gb",
                "disk_free_gb",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    elif mode == "bin":
        with open(output_file, "wb") as binfile:
            for record in records:
                packed = struct.pack(
                    RECORD_FORMAT_INTERVAL,
                    record["seconds"],
                    record["cpu_percent"],
                    record["ram_available_gb"],
                    record["disk_free_gb"],
                )
                binfile.write(packed)
    else:
        print("Unsupported mode. Please select 'csv' or 'json'.")
        return
    print(f"Interval metrics written to {output_file}")


def convert_global_bin(file_path, mode="csv"):
    """Convert the global system metrics binary file to CSV or JSON."""
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "rb") as f:
        record = f.read(RECORD_SIZE_GLOBAL)
        if not record:
            print("Global file is empty.")
            return
        total_ram_gb, disk_total_gb, disk_start_free_gb = struct.unpack(
            RECORD_FORMAT_GLOBAL, record
        )
        global_metrics = {
            "total_ram_gb": total_ram_gb,
            "disk_total_gb": disk_total_gb,
            "disk_start_free_gb": disk_start_free_gb,
        }

    output_file = file_path.replace(".bin", f"_converted.{mode}")
    write_global(output_file, global_metrics, mode)


def convert_interval_bin(file_path, preview=False, mode="csv"):
    """Convert the interval system metrics binary file to CSV or JSON."""
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    total_records = 0
    processed_records = []

    with open(file_path, "rb") as f:
        while True:
            record = f.read(RECORD_SIZE_INTERVAL)
            if not record:
                break
            seconds, cpu, ram, disk = struct.unpack(RECORD_FORMAT_INTERVAL, record)
            processed_records.append(
                {
                    "seconds": seconds,
                    "cpu_percent": cpu,
                    "ram_available_gb": ram,
                    "disk_free_gb": disk,
                }
            )
            total_records += 1
            if preview and total_records >= 5:
                break

    output_file = file_path.replace(".bin", f"_converted.{mode}")
    if preview:
        with open(output_file, "w") as jsonfile:
            json.dump(processed_records, jsonfile, indent=4)
        estimated_size = (
            os.path.getsize(output_file)
            / 5
            * (os.path.getsize(file_path) - RECORD_SIZE_GLOBAL)
            / RECORD_SIZE_INTERVAL
        )
        print(f"Estimated converted file size: ~{size_formating(int(estimated_size))}")
        os.remove(output_file)
    else:
        write_interval(output_file, processed_records, mode)


def size_formating(estimated_size):
    if estimated_size >= 1000000000:
        return f"{round(estimated_size / 1000000000, 3)}gb"
    elif estimated_size >= 1000000:
        return f"{round(estimated_size / 1000000, 3)}mb"
    elif estimated_size >= 1000:
        return f"{round(estimated_size / 1000, 3)}kb"
    else:
        return f"{estimated_size}b"
