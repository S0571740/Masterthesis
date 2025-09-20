import argparse
import subprocess
import threading
import Logger
import os
import logging
import time
import json
import TokenRateLimiter
import shutil
from Converter import convert_bin_to_file
from Converter import merge
from Converter import final_merge
from MSGrag import ms_build_kg
from MSGrag import query_grag as ms_grag
from NeoGrag import query_grag as neo_grag
from NeoGrag import createGRAG as neo_build_kg

stats_path = "MSGrag/output/stats.json"
creds_file = "NeoGrag/neo4j_creds.json"


def write_to_file(text, file):
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n"))


def copy_txt_files(src_dir, dst_dir):
    """
    Copy all .txt files from src_dir to dst_dir.
    Only .txt files are copied. Other files are ignored.
    """
    os.makedirs(dst_dir, exist_ok=True)  # ensure destination exists

    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        if os.path.isfile(src_path) and file_name.lower().endswith(".txt"):
            dst_path = os.path.join(dst_dir, file_name)
            shutil.copy2(src_path, dst_path)  # copy preserving metadata


def copy_txt_from_subdirs(src_root, dst_root):
    """
    Iterate over all subdirectories in src_root and copy .txt files
    into corresponding directories in dst_root.
    """
    os.makedirs(dst_root, exist_ok=True)

    for entry in os.listdir(src_root):
        src_path = os.path.join(src_root, entry)
        if os.path.isdir(src_path):
            # Destination subdirectory
            dst_path = os.path.join(dst_root, entry)
            # Copy .txt files
            copy_txt_files(src_path, dst_path)


def run_monitor(log_path="metrics"):
    running_flag = {"running": True}
    logger = Logger.Logger(log_path)
    monitor_thread = threading.Thread(
        target=Logger.monitor_system, args=(logger, 1, running_flag)
    )
    monitor_thread.start()
    return monitor_thread, running_flag


def args_not_valid(output_dir, questions_dir, input_path):
    if not output_dir or not questions_dir:
        print("Error: --output_dir and --questions_dir are required for run_all mode.")
        return True
    if not os.path.exists(input_path):
        print(
            "Error: input path for files to build graph from is empty/does not exist!"
        )
        return True
    return False


def number_of_files_in_dir(dir_path):
    return len(
        [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    )


def knowledge_graph_already_present(system, files_count, force_rebuild):
    if system == "ms":
        if not force_rebuild and (
            not os.path.exists(stats_path)
            or not json.load(open(stats_path, "r", encoding="utf-8"))["num_documents"]
            == files_count
        ):
            return False
        return True
    elif system == "neo":
        if not force_rebuild and (
            not os.path.exists(creds_file)
            or not json.load(open(creds_file, "r", encoding="utf-8"))["num_documents"]
            == files_count
        ):
            return False
        return True


def build_kg(system, rate_limiter):
    if system == "ms":
        monitor_thread_kg_ms, running_flag_kg_ms = run_monitor("kg_ms_metrics")
        try:
            ms_build_kg()
        except subprocess.CalledProcessError as e:
            print(f"ms grag failed: {e}")
        finally:
            running_flag_kg_ms["running"] = False
            monitor_thread_kg_ms.join()
            print(f"System metrics logged to kg_ms_metrics")
            time.sleep(60)
    elif system == "neo":
        metrics_kg_neo = "kg_neo_metrics"
        monitor_thread_kg_neo, running_flag_kg_neo = run_monitor(metrics_kg_neo)
        try:
            neo_build_kg(rate_limiter)
        except subprocess.CalledProcessError as e:
            print(f"neo grag failed: {e}")
        finally:
            running_flag_kg_neo["running"] = False
            monitor_thread_kg_neo.join()
            print(f"System metrics logged to {metrics_kg_neo}")
            time.sleep(60)


def setup_directories(output_dir, questions_dir):
    output_dir = os.path.abspath(output_dir)
    questions_dir = os.path.abspath(questions_dir)
    ms_output_dir = os.path.join(output_dir, "ms")
    neo_output_dir = os.path.join(output_dir, "neo")
    return output_dir, questions_dir, ms_output_dir, neo_output_dir


def querry_system(system, current_output_dir, index, question, locality="local"):
    if system == "ms":
        metrics_query_ms = os.path.join(
            current_output_dir, f"metrics_{locality}_{index}"
        )
        monitor_thread_query_ms, running_flag_query_ms = run_monitor(metrics_query_ms)
        try:
            response = ms_grag(query=question, type=locality)
        except subprocess.CalledProcessError as e:
            print(f"ms grag failed: {e}")
        finally:
            running_flag_query_ms["running"] = False
            monitor_thread_query_ms.join()
            write_to_file(
                response, f"{current_output_dir}/response_{locality}_ms_{index}.txt"
            )
    elif system == "neo":
        metrics_query_neo = os.path.join(
            current_output_dir, f"metrics_{locality}_{index}"
        )
        monitor_thread_query_neo, running_flag_query_neo = run_monitor(
            metrics_query_neo
        )
        try:
            response = neo_grag(query=question, type=locality)
        except subprocess.CalledProcessError as e:
            print(f"neo grag failed: {e}")
        finally:
            running_flag_query_neo["running"] = False
            monitor_thread_query_neo.join()
            write_to_file(
                response, f"{current_output_dir}/response_{locality}_neo_{index}.txt"
            )


def get_output_dirs(base_ms, base_neo, question_file):
    name = os.path.splitext(os.path.basename(question_file))[0]
    return (
        os.path.join(base_ms, name),
        os.path.join(base_neo, name),
    )


def main():
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(
        description="Hardware Monitor and GraphRAG Wrapper for Microsoft and Neo4J Solutions"
    )
    parser.add_argument(
        "mode",
        choices=["convert", "run_all"],
        help="Mode to run the project.",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the binary file for conversion (required for convert mode).",
    )
    parser.add_argument(
        "--build_kg",
        type=str,
        help="Path to the file(s) to be used for creating a knowledge graph.",
    )
    parser.add_argument(
        "-c",
        "--convert_mode",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Conversion output format.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test mode: preview and estimate file size.",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        help="Output directory for result files (used in run_all mode).",
    )
    parser.add_argument(
        "-qd",
        "--questions_dir",
        type=str,
        help="Directory containing question files (used in run_all mode).",
    )

    args = parser.parse_args()

    if args.mode == "convert":
        if not args.path:
            print(
                "Error: You must provide a path to the binary log file using --path when using convert mode."
            )
            return
        convert_bin_to_file(args.path, mode=args.convert_mode, preview=args.test)
    elif args.mode == "run_all":
        input_path = "MSGrag/input"
        rate_limiter = TokenRateLimiter.TokenRateLimiter(max_tokens_per_minute=10_000)
        if args_not_valid(args.output_dir, args.questions_dir, input_path):
            return

        # Build MS KG
        input_files_count = number_of_files_in_dir(input_path)
        if knowledge_graph_already_present("ms", input_files_count, True):
            build_kg("ms", rate_limiter)
            print("MS KG was freshly build")
        else:
            print("MS KG was already present")
        # Build Neo KG
        if knowledge_graph_already_present("neo", input_files_count, True):
            build_kg("neo", rate_limiter)
            print("Neo KG was freshly build")
        else:
            print("Neo KG was already present")

        output_dir, questions_dir, ms_output_dir, neo_output_dir = setup_directories(
            args.output_dir, args.questions_dir
        )

        q_files_count = number_of_files_in_dir(input_path)

        print("Starting Neo4J Q&A")
        for counter, questions_file in enumerate(os.listdir(questions_dir), start=1):
            print(f"{counter}/{q_files_count} Handling Q&A for: {questions_file}")

            question_file_path = os.path.join(questions_dir, questions_file)
            with open(question_file_path, "r", encoding="utf-8") as questions:
                current_ms_output_dir, current_neo_output_dir = get_output_dirs(
                    ms_output_dir, neo_output_dir, question_file_path
                )
                for index, question in enumerate(questions, start=1):
                    if question.strip():
                        querry_system(
                            "neo",
                            current_neo_output_dir,
                            index,
                            question.strip(),
                            "local",
                        )
                        # querry_system(
                        #     "neo",
                        #     current_neo_output_dir,
                        #     index,
                        #     question.strip(),
                        #     "global",
                        # )
                merge(
                    current_neo_output_dir,
                    os.path.join(neo_output_dir, questions_file),
                    "bin",
                )
            print(f"{counter}/{q_files_count} Done with: {questions_file}")

        print("Starting MS Q&A")
        for counter, questions_file in enumerate(os.listdir(questions_dir), start=1):
            print(f"{counter}/{q_files_count} Handling Q&A for: {questions_file}")

            question_file_path = os.path.join(questions_dir, questions_file)
            with open(question_file_path, "r", encoding="utf-8") as questions:
                current_ms_output_dir, current_neo_output_dir = get_output_dirs(
                    ms_output_dir, neo_output_dir, question_file_path
                )
                for index, question in enumerate(questions, start=1):
                    print(f"index: {index}")
                    if question.strip():
                        querry_system(
                            "ms",
                            current_ms_output_dir,
                            index,
                            question.strip(),
                            "local",
                        )
                        # querry_system(
                        #     "ms",
                        #     current_ms_output_dir,
                        #     index,
                        #     question.strip(),
                        #     "global",
                        # )
                merge(
                    current_ms_output_dir,
                    os.path.join(ms_output_dir, questions_file),
                    "bin",
                )
        final_merge(output_dir)
    else:
        print("Invalid mode selected.")


if __name__ == "__main__":
    main()
