import csv
import statistics
import json

def calc_stats(group):
    stats = {}
    for col, values in group.items():
        stats[col] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "q25": (
                statistics.quantiles(values, n=4, method='inclusive')[0] if len(values) > 1 else values[0]
            ),
            "q75": (
                statistics.quantiles(values, n=4, method='inclusive')[2] if len(values) > 1 else values[0]
            ),
        }
    return stats

def create_grouped_stats(rows, batch_size=25, split_marker=0):
    every_groups = []
    file_groups = []
    current_subgroup = {k: [] for k in rows[0].keys()}
    subgroup_stats_list = []
    subgroup_counter = 0

    for row in rows:
        # Check for split_marker
        if row["seconds"] == split_marker and current_subgroup["seconds"]:
            # Compute stats for subgroup
            stats_subgroup = calc_stats(current_subgroup)
            every_groups.append(stats_subgroup)
            subgroup_stats_list.append(stats_subgroup)
            subgroup_counter += 1

            # If batch_size subgroups collected, compute overarching stats
            if subgroup_counter % batch_size == 0:
                batch_stats = {}
                for col in current_subgroup:
                    col_means = [s[col]["mean"] for s in subgroup_stats_list]
                    col_medians = [s[col]["median"] for s in subgroup_stats_list]
                    col_mins = [s[col]["min"] for s in subgroup_stats_list]
                    col_maxs = [s[col]["max"] for s in subgroup_stats_list]

                    batch_stats[col] = {
                        "mean": statistics.mean(col_means),
                        "median": statistics.median(col_medians),
                        "stdev": statistics.stdev(col_means) if len(col_means) > 1 else 0,
                        "min": min(col_mins),
                        "max": max(col_maxs),
                        "count": len(col_means),
                        "q25": statistics.quantiles(col_medians, n=4)[0] if len(col_medians) > 1 else col_medians[0],
                        "q75": statistics.quantiles(col_medians, n=4)[2] if len(col_medians) > 1 else col_medians[0],
                    }
                file_groups.append(batch_stats)
                subgroup_stats_list = []

            # Start a new subgroup
            current_subgroup = {k: [] for k in current_subgroup}

        # Add row to current subgroup
        for k in current_subgroup:
            current_subgroup[k].append(row[k])

    # Finalize last subgroup if not empty
    if current_subgroup["seconds"]:
        stats_subgroup = calc_stats(current_subgroup)
        every_groups.append(stats_subgroup)
        subgroup_stats_list.append(stats_subgroup)

    # Compute final batch if any remaining subgroups
    if subgroup_stats_list:
        batch_stats = {}
        for col in current_subgroup:
            col_means = [s[col]["mean"] for s in subgroup_stats_list]
            col_medians = [s[col]["median"] for s in subgroup_stats_list]
            col_mins = [s[col]["min"] for s in subgroup_stats_list]
            col_maxs = [s[col]["max"] for s in subgroup_stats_list]

            batch_stats[col] = {
                "mean": statistics.mean(col_means),
                "median": statistics.median(col_medians),
                "stdev": statistics.stdev(col_means) if len(col_means) > 1 else 0,
                "min": min(col_mins),
                "max": max(col_maxs),
                "count": len(col_means),
                "q25": statistics.quantiles(col_medians, n=4)[0] if len(col_medians) > 1 else col_medians[0],
                "q75": statistics.quantiles(col_medians, n=4)[2] if len(col_medians) > 1 else col_medians[0],
            }
        file_groups.append(batch_stats)

    return file_groups, every_groups

# Hardcoded CSV path (change to your file)

path = "TODO"
ms_csv_file = f"{path}/final_merge_ms_INTERVAL.csv"
neo_csv_file = f"{path}/final_merge_neo_INTERVAL.csv"

ms_rows = []
neo_rows = []

# Read CSV into a list of rows
with open(ms_csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ms_rows.append(
            {
                "seconds": int(row["seconds"]),
                "cpu_percent": float(row["cpu_percent"]),
                "ram_available_gb": float(row["ram_available_gb"]),
                "disk_free_gb": float(row["disk_free_gb"]),
            }
        )

with open(neo_csv_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        neo_rows.append(
            {
                "seconds": int(row["seconds"]),
                "cpu_percent": float(row["cpu_percent"]),
                "ram_available_gb": float(row["ram_available_gb"]),
                "disk_free_gb": float(row["disk_free_gb"]),
            }
        )

# MS data
ms_grouped_groups, ms_each_groups = create_grouped_stats(ms_rows, 25)
# Neo data
neo_grouped_groups, neo_each_groups = create_grouped_stats(neo_rows, 25)

with open("ms_each.json", "w", encoding="utf-8") as f:
    json.dump(ms_each_groups, f, indent=4)

with open("ms_grouped.json", "w", encoding="utf-8") as f:
    json.dump(ms_grouped_groups, f, indent=4)


with open("neo_each.json", "w", encoding="utf-8") as f:
    json.dump(neo_each_groups, f, indent=4)

with open("neo_grouped.json", "w", encoding="utf-8") as f:
    json.dump(neo_grouped_groups, f, indent=4)
