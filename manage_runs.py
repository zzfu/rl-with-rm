"""
Interactive run management script.

List, inspect, and delete training runs (checkpoints and logs).
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_runs(checkpoint_dir: str = "./checkpoints/rm", log_dir: str = "./logs/rm") -> dict:
    """
    Scan checkpoint and log directories to find all runs.

    Returns:
        dict mapping run_name -> {"checkpoints": Path|None, "logs": Path|None, ...}
    """
    runs = {}

    checkpoint_path = Path(checkpoint_dir)
    log_path = Path(log_dir)

    # Scan checkpoints
    if checkpoint_path.exists():
        for run_dir in checkpoint_path.iterdir():
            if run_dir.is_dir():
                run_name = run_dir.name
                runs[run_name] = {
                    "checkpoints": run_dir,
                    "logs": None,
                    "checkpoint_size": get_dir_size(run_dir),
                    "log_size": 0,
                }

    # Scan logs
    if log_path.exists():
        for run_dir in log_path.iterdir():
            if run_dir.is_dir():
                run_name = run_dir.name
                if run_name in runs:
                    runs[run_name]["logs"] = run_dir
                    runs[run_name]["log_size"] = get_dir_size(run_dir)
                else:
                    runs[run_name] = {
                        "checkpoints": None,
                        "logs": run_dir,
                        "checkpoint_size": 0,
                        "log_size": get_dir_size(run_dir),
                    }

    # Add total size and modified time
    for run_name, info in runs.items():
        info["total_size"] = info["checkpoint_size"] + info["log_size"]

        # Get most recent modification time
        mtime = 0
        if info["checkpoints"]:
            mtime = max(mtime, info["checkpoints"].stat().st_mtime)
        if info["logs"]:
            mtime = max(mtime, info["logs"].stat().st_mtime)
        info["mtime"] = mtime

    return runs


def list_runs(runs: dict, marked: set) -> None:
    """Display all runs with their info."""
    if not runs:
        print("No runs found.")
        return

    # Sort by modification time (newest first)
    sorted_runs = sorted(runs.items(), key=lambda x: x[1]["mtime"], reverse=True)

    print("\n" + "=" * 70)
    print(f"{'#':<4} {'Run Name':<25} {'Size':<10} {'Ckpt':<5} {'Logs':<5} {'Del?':<5}")
    print("=" * 70)

    for i, (run_name, info) in enumerate(sorted_runs, 1):
        ckpt = "✓" if info["checkpoints"] else "-"
        logs = "✓" if info["logs"] else "-"
        marked_str = "[X]" if run_name in marked else "[ ]"
        size_str = format_size(info["total_size"])

        # Truncate run name if too long
        display_name = run_name[:24] if len(run_name) > 24 else run_name

        print(f"{i:<4} {display_name:<25} {size_str:<10} {ckpt:<5} {logs:<5} {marked_str:<5}")

    print("=" * 70)
    print(f"Total: {len(runs)} runs, {format_size(sum(r['total_size'] for r in runs.values()))}")
    if marked:
        marked_size = sum(runs[r]["total_size"] for r in marked if r in runs)
        print(f"Marked for deletion: {len(marked)} runs, {format_size(marked_size)}")
    print()


def interactive_session(runs: dict) -> set:
    """Interactive session to mark runs for deletion."""
    marked = set()
    sorted_run_names = sorted(runs.keys(), key=lambda x: runs[x]["mtime"], reverse=True)

    print("\nCommands:")
    print("  <number>  - Toggle mark for deletion")
    print("  a         - Mark all")
    print("  n         - Unmark all")
    print("  l         - List runs")
    print("  d         - Done, proceed to deletion")
    print("  q         - Quit without deleting")
    print()

    list_runs(runs, marked)

    while True:
        try:
            cmd = input("Enter command: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return set()

        if cmd == "q":
            print("Aborted.")
            return set()
        elif cmd == "d":
            break
        elif cmd == "l":
            list_runs(runs, marked)
        elif cmd == "a":
            marked = set(runs.keys())
            list_runs(runs, marked)
        elif cmd == "n":
            marked = set()
            list_runs(runs, marked)
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(sorted_run_names):
                run_name = sorted_run_names[idx]
                if run_name in marked:
                    marked.remove(run_name)
                    print(f"Unmarked: {run_name}")
                else:
                    marked.add(run_name)
                    print(f"Marked: {run_name}")
            else:
                print(f"Invalid number. Enter 1-{len(runs)}")
        else:
            print("Unknown command. Enter number, 'a', 'n', 'l', 'd', or 'q'")

    return marked


def delete_runs(runs: dict, to_delete: set) -> None:
    """Delete the specified runs."""
    for run_name in to_delete:
        info = runs[run_name]
        print(f"Deleting {run_name}...", end=" ")

        try:
            if info["checkpoints"] and info["checkpoints"].exists():
                shutil.rmtree(info["checkpoints"])
            if info["logs"] and info["logs"].exists():
                shutil.rmtree(info["logs"])
            print("done")
        except Exception as e:
            print(f"error: {e}")


def main():
    print("Run Management Tool")
    print("=" * 40)

    runs = get_runs()

    if not runs:
        print("No runs found in ./checkpoints/rm or ./logs/rm")
        return

    marked = interactive_session(runs)

    if not marked:
        print("No runs marked for deletion.")
        return

    # Final confirmation
    print("\n" + "=" * 50)
    print("RUNS TO BE DELETED:")
    print("=" * 50)

    total_size = 0
    for run_name in sorted(marked):
        info = runs[run_name]
        total_size += info["total_size"]
        print(f"  {run_name} ({format_size(info['total_size'])})")
        if info["checkpoints"]:
            print(f"    - {info['checkpoints']}")
        if info["logs"]:
            print(f"    - {info['logs']}")

    print("=" * 50)
    print(f"Total: {len(marked)} runs, {format_size(total_size)}")
    print()

    try:
        confirm = input("Type 'DELETE' to confirm deletion: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    if confirm == "DELETE":
        delete_runs(runs, marked)
        print(f"\nDeleted {len(marked)} runs.")
    else:
        print("Aborted. Nothing was deleted.")


if __name__ == "__main__":
    main()
