"""
Interactive run management script.

List, inspect, and delete training runs (checkpoints and logs).
Supports both RM and GRPO runs, and snapshot-level deletion.
"""

import shutil
from pathlib import Path


# Directories to scan
RUN_DIRS = [
    {"type": "RM", "checkpoints": "./checkpoints/rm", "logs": "./logs/rm"},
    {"type": "GRPO", "checkpoints": "./checkpoints/grpo", "logs": "./logs/grpo"},
]


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


def get_runs() -> dict:
    """
    Scan checkpoint and log directories to find all runs.

    Returns:
        dict mapping run_name -> {"checkpoints": Path|None, "logs": Path|None, "type": str, ...}
    """
    runs = {}

    for run_dir_config in RUN_DIRS:
        run_type = run_dir_config["type"]
        checkpoint_path = Path(run_dir_config["checkpoints"])
        log_path = Path(run_dir_config["logs"])

        # Scan checkpoints
        if checkpoint_path.exists():
            for run_dir in checkpoint_path.iterdir():
                if run_dir.is_dir():
                    run_name = run_dir.name
                    # Use composite key to avoid name collisions between RM and GRPO
                    key = f"{run_type}:{run_name}"
                    runs[key] = {
                        "name": run_name,
                        "type": run_type,
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
                    key = f"{run_type}:{run_name}"
                    if key in runs:
                        runs[key]["logs"] = run_dir
                        runs[key]["log_size"] = get_dir_size(run_dir)
                    else:
                        runs[key] = {
                            "name": run_name,
                            "type": run_type,
                            "checkpoints": None,
                            "logs": run_dir,
                            "checkpoint_size": 0,
                            "log_size": get_dir_size(run_dir),
                        }

    # Add total size and modified time
    for key, info in runs.items():
        info["total_size"] = info["checkpoint_size"] + info["log_size"]

        # Get most recent modification time
        mtime = 0
        if info["checkpoints"]:
            mtime = max(mtime, info["checkpoints"].stat().st_mtime)
        if info["logs"]:
            mtime = max(mtime, info["logs"].stat().st_mtime)
        info["mtime"] = mtime

    return runs


def get_snapshots(run_info: dict) -> list:
    """Get list of snapshots (step-X directories) for a run."""
    snapshots = []
    if run_info["checkpoints"] and run_info["checkpoints"].exists():
        for item in run_info["checkpoints"].iterdir():
            if item.is_dir() and item.name.startswith("step-"):
                try:
                    step_num = int(item.name.split("-")[1])
                    snapshots.append({
                        "name": item.name,
                        "path": item,
                        "step": step_num,
                        "size": get_dir_size(item),
                    })
                except (ValueError, IndexError):
                    pass
    # Sort by step number
    snapshots.sort(key=lambda x: x["step"])
    return snapshots


def list_runs(runs: dict, marked: set) -> list:
    """Display all runs with their info. Returns sorted keys."""
    if not runs:
        print("No runs found.")
        return []

    # Sort by modification time (newest first)
    sorted_items = sorted(runs.items(), key=lambda x: x[1]["mtime"], reverse=True)
    sorted_keys = [k for k, _ in sorted_items]

    print("\n" + "=" * 80)
    print(f"{'#':<4} {'Type':<6} {'Run Name':<30} {'Size':<10} {'Ckpt':<5} {'Logs':<5} {'Del?':<5}")
    print("=" * 80)

    for i, (key, info) in enumerate(sorted_items, 1):
        ckpt = "Y" if info["checkpoints"] else "-"
        logs = "Y" if info["logs"] else "-"
        marked_str = "[X]" if key in marked else "[ ]"
        size_str = format_size(info["total_size"])

        # Truncate run name if too long
        display_name = info["name"][:29] if len(info["name"]) > 29 else info["name"]

        print(f"{i:<4} {info['type']:<6} {display_name:<30} {size_str:<10} {ckpt:<5} {logs:<5} {marked_str:<5}")

    print("=" * 80)
    print(f"Total: {len(runs)} runs, {format_size(sum(r['total_size'] for r in runs.values()))}")
    if marked:
        marked_size = sum(runs[r]["total_size"] for r in marked if r in runs)
        print(f"Marked for deletion: {len(marked)} runs, {format_size(marked_size)}")
    print()

    return sorted_keys


def list_snapshots(run_info: dict, snapshots: list, marked: set) -> None:
    """Display snapshots for a run."""
    print(f"\nSnapshots for '{run_info['name']}' ({run_info['type']}):")
    print("-" * 50)

    if not snapshots:
        print("  No snapshots found.")
        return

    for i, snap in enumerate(snapshots, 1):
        marked_str = "[X]" if snap["name"] in marked else "[ ]"
        print(f"  {i:<4} {snap['name']:<20} {format_size(snap['size']):<10} {marked_str}")

    print("-" * 50)
    if marked:
        marked_size = sum(s["size"] for s in snapshots if s["name"] in marked)
        print(f"Marked: {len(marked)} snapshots, {format_size(marked_size)}")
    print()


def snapshot_session(run_info: dict) -> set:
    """Interactive session to mark snapshots for deletion."""
    snapshots = get_snapshots(run_info)
    if not snapshots:
        print(f"No snapshots found for '{run_info['name']}'")
        return set()

    marked = set()
    snapshot_names = [s["name"] for s in snapshots]

    print("\nSnapshot commands:")
    print("  <number>  - Toggle mark for deletion")
    print("  a         - Mark all")
    print("  n         - Unmark all")
    print("  l         - List snapshots")
    print("  d         - Done, delete marked snapshots")
    print("  b         - Back (cancel)")
    print()

    list_snapshots(run_info, snapshots, marked)

    while True:
        try:
            cmd = input("Snapshot> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return set()

        if cmd == "b":
            return set()
        elif cmd == "d":
            break
        elif cmd == "l":
            list_snapshots(run_info, snapshots, marked)
        elif cmd == "a":
            marked = set(snapshot_names)
            list_snapshots(run_info, snapshots, marked)
        elif cmd == "n":
            marked = set()
            list_snapshots(run_info, snapshots, marked)
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(snapshots):
                name = snapshots[idx]["name"]
                if name in marked:
                    marked.remove(name)
                    print(f"Unmarked: {name}")
                else:
                    marked.add(name)
                    print(f"Marked: {name}")
            else:
                print(f"Invalid number. Enter 1-{len(snapshots)}")
        else:
            print("Unknown command.")

    return marked


def delete_snapshots(run_info: dict, snapshots: list, to_delete: set) -> None:
    """Delete marked snapshots."""
    for snap in snapshots:
        if snap["name"] in to_delete:
            print(f"  Deleting {snap['name']}...", end=" ")
            try:
                shutil.rmtree(snap["path"])
                print("done")
            except Exception as e:
                print(f"error: {e}")


def interactive_session(runs: dict) -> tuple:
    """Interactive session to mark runs for deletion. Returns (marked_runs, snapshot_deletions)."""
    marked = set()
    snapshot_deletions = {}  # key -> set of snapshot names

    print("\nCommands:")
    print("  <number>   - Toggle mark run for deletion")
    print("  s<number>  - Manage snapshots (e.g., s1, s2)")
    print("  a          - Mark all runs")
    print("  n          - Unmark all runs")
    print("  l          - List runs")
    print("  d          - Done, proceed to deletion")
    print("  q          - Quit without deleting")
    print()

    sorted_keys = list_runs(runs, marked)

    while True:
        try:
            cmd = input("Enter command: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return set(), {}

        if cmd == "q":
            print("Aborted.")
            return set(), {}
        elif cmd == "d":
            break
        elif cmd == "l":
            sorted_keys = list_runs(runs, marked)
        elif cmd == "a":
            marked = set(runs.keys())
            list_runs(runs, marked)
        elif cmd == "n":
            marked = set()
            list_runs(runs, marked)
        elif cmd.startswith("s") and cmd[1:].isdigit():
            idx = int(cmd[1:]) - 1
            if 0 <= idx < len(sorted_keys):
                key = sorted_keys[idx]
                run_info = runs[key]
                if not run_info["checkpoints"]:
                    print(f"No checkpoints for '{run_info['name']}'")
                    continue
                snapshots_to_delete = snapshot_session(run_info)
                if snapshots_to_delete:
                    snapshot_deletions[key] = snapshots_to_delete
                    print(f"Queued {len(snapshots_to_delete)} snapshots for deletion")
            else:
                print(f"Invalid number. Enter s1-s{len(runs)}")
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(sorted_keys):
                key = sorted_keys[idx]
                if key in marked:
                    marked.remove(key)
                    print(f"Unmarked: {runs[key]['name']}")
                else:
                    marked.add(key)
                    print(f"Marked: {runs[key]['name']}")
            else:
                print(f"Invalid number. Enter 1-{len(runs)}")
        else:
            print("Unknown command. Enter number, s<number>, 'a', 'n', 'l', 'd', or 'q'")

    return marked, snapshot_deletions


def delete_runs(runs: dict, to_delete: set) -> None:
    """Delete the specified runs."""
    for key in to_delete:
        info = runs[key]
        print(f"Deleting {info['type']}:{info['name']}...", end=" ")

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
        print("No runs found in checkpoint/log directories.")
        return

    marked, snapshot_deletions = interactive_session(runs)

    if not marked and not snapshot_deletions:
        print("Nothing marked for deletion.")
        return

    # Final confirmation
    print("\n" + "=" * 60)
    print("ITEMS TO BE DELETED:")
    print("=" * 60)

    total_size = 0

    # Show runs to delete
    if marked:
        print("\nFULL RUNS:")
        for key in sorted(marked):
            info = runs[key]
            total_size += info["total_size"]
            print(f"  [{info['type']}] {info['name']} ({format_size(info['total_size'])})")
            if info["checkpoints"]:
                print(f"      {info['checkpoints']}")
            if info["logs"]:
                print(f"      {info['logs']}")

    # Show snapshots to delete
    if snapshot_deletions:
        print("\nSNAPSHOTS:")
        for key, snap_names in snapshot_deletions.items():
            info = runs[key]
            snapshots = get_snapshots(info)
            snap_size = sum(s["size"] for s in snapshots if s["name"] in snap_names)
            total_size += snap_size
            print(f"  [{info['type']}] {info['name']}:")
            for snap in snapshots:
                if snap["name"] in snap_names:
                    print(f"      {snap['name']} ({format_size(snap['size'])})")

    print("=" * 60)
    count = len(marked) + sum(len(s) for s in snapshot_deletions.values())
    print(f"Total: {count} items, {format_size(total_size)}")
    print()

    try:
        confirm = input("Type 'DELETE' to confirm deletion: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return

    if confirm == "DELETE":
        # Delete snapshots first
        for key, snap_names in snapshot_deletions.items():
            info = runs[key]
            snapshots = get_snapshots(info)
            print(f"Deleting snapshots from {info['name']}:")
            delete_snapshots(info, snapshots, snap_names)

        # Delete full runs
        if marked:
            print("Deleting full runs:")
            delete_runs(runs, marked)

        print(f"\nDeletion complete.")
    else:
        print("Aborted. Nothing was deleted.")


if __name__ == "__main__":
    main()
