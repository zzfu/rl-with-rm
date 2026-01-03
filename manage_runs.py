"""
Interactive run management script.

List, inspect, and delete training runs (checkpoints and logs).
Supports both RM and GRPO runs, and snapshot-level deletion.
"""

import shutil
from pathlib import Path


# Directories to scan
RUN_DIRS = [
    {"type": "RM", "checkpoints": "./checkpoints/rm", "logs": "./logs/rm", "rollouts": None},
    {"type": "GRPO", "checkpoints": "./checkpoints/grpo", "logs": "./logs/grpo", "rollouts": "./rollouts"},
    {"type": "QUANT", "checkpoints": "./quantized_models", "logs": None, "rollouts": None},
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
        dict mapping run_name -> {"checkpoints": Path|None, "logs": Path|None, "rollouts": Path|None, "type": str, ...}
    """
    runs = {}

    for run_dir_config in RUN_DIRS:
        run_type = run_dir_config["type"]
        checkpoint_path = Path(run_dir_config["checkpoints"])
        log_path = Path(run_dir_config["logs"]) if run_dir_config["logs"] else None
        rollouts_path = Path(run_dir_config["rollouts"]) if run_dir_config["rollouts"] else None

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
                        "rollouts": None,
                        "checkpoint_size": get_dir_size(run_dir),
                        "log_size": 0,
                        "rollouts_size": 0,
                    }

        # Scan logs
        if log_path and log_path.exists():
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
                            "rollouts": None,
                            "checkpoint_size": 0,
                            "log_size": get_dir_size(run_dir),
                            "rollouts_size": 0,
                        }

        # Scan rollouts
        if rollouts_path and rollouts_path.exists():
            for run_dir in rollouts_path.iterdir():
                if run_dir.is_dir():
                    run_name = run_dir.name
                    key = f"{run_type}:{run_name}"
                    if key in runs:
                        runs[key]["rollouts"] = run_dir
                        runs[key]["rollouts_size"] = get_dir_size(run_dir)
                    else:
                        runs[key] = {
                            "name": run_name,
                            "type": run_type,
                            "checkpoints": None,
                            "logs": None,
                            "rollouts": run_dir,
                            "checkpoint_size": 0,
                            "log_size": 0,
                            "rollouts_size": get_dir_size(run_dir),
                        }

    # Add total size and modified time
    for key, info in runs.items():
        info["total_size"] = info["checkpoint_size"] + info["log_size"] + info["rollouts_size"]

        # Get most recent modification time
        mtime = 0
        if info["checkpoints"]:
            mtime = max(mtime, info["checkpoints"].stat().st_mtime)
        if info["logs"]:
            mtime = max(mtime, info["logs"].stat().st_mtime)
        if info["rollouts"]:
            mtime = max(mtime, info["rollouts"].stat().st_mtime)
        info["mtime"] = mtime

    return runs


def get_quantized_subfolders(snapshot_path: Path) -> list:
    """Get list of quantized subfolders within a snapshot."""
    quantized = []
    if snapshot_path.exists():
        for item in snapshot_path.iterdir():
            if item.is_dir() and item.name.startswith("quantized_"):
                quant_type = item.name.replace("quantized_", "")
                quantized.append({
                    "name": item.name,
                    "path": item,
                    "type": quant_type,
                    "size": get_dir_size(item),
                })
    # Sort by name
    quantized.sort(key=lambda x: x["name"])
    return quantized


def get_snapshots(run_info: dict) -> list:
    """Get list of snapshots (step-X directories) for a run."""
    snapshots = []
    if run_info["checkpoints"] and run_info["checkpoints"].exists():
        for item in run_info["checkpoints"].iterdir():
            if item.is_dir() and item.name.startswith("step-"):
                try:
                    step_num = int(item.name.split("-")[1])
                    quantized = get_quantized_subfolders(item)
                    snapshots.append({
                        "name": item.name,
                        "path": item,
                        "step": step_num,
                        "size": get_dir_size(item),
                        "quantized": quantized,
                    })
                except (ValueError, IndexError):
                    pass
    # Sort by step number
    snapshots.sort(key=lambda x: x["step"])
    return snapshots


def list_runs(runs: dict, marked: set, snapshot_deletions: dict = None) -> list:
    """Display all runs with their info. Returns sorted keys."""
    snapshot_deletions = snapshot_deletions or {}
    if not runs:
        print("No runs found.")
        return []

    # Sort by modification time (newest first)
    sorted_items = sorted(runs.items(), key=lambda x: x[1]["mtime"], reverse=True)
    sorted_keys = [k for k, _ in sorted_items]

    print("\n" + "=" * 85)
    print(f"{'#':<4} {'Type':<6} {'Run Name':<30} {'Size':<10} {'Ckpt':<5} {'Logs':<5} {'Roll':<5} {'Del?':<5}")
    print("=" * 85)

    for i, (key, info) in enumerate(sorted_items, 1):
        ckpt = "Y" if info["checkpoints"] else "-"
        logs = "Y" if info["logs"] else "-"
        roll = "Y" if info.get("rollouts") else "-"
        # [X] = full run, [~] = some snapshots, [ ] = nothing
        if key in marked:
            marked_str = "[X]"
        elif key in snapshot_deletions:
            marked_str = "[~]"
        else:
            marked_str = "[ ]"
        size_str = format_size(info["total_size"])

        # Truncate run name if too long
        display_name = info["name"][:29] if len(info["name"]) > 29 else info["name"]

        print(f"{i:<4} {info['type']:<6} {display_name:<30} {size_str:<10} {ckpt:<5} {logs:<5} {roll:<5} {marked_str:<5}")

    print("=" * 85)
    print(f"Total: {len(runs)} runs, {format_size(sum(r['total_size'] for r in runs.values()))}")
    if marked:
        marked_size = sum(runs[r]["total_size"] for r in marked if r in runs)
        print(f"Marked for deletion: {len(marked)} runs, {format_size(marked_size)}")
    print()

    return sorted_keys


def build_snapshot_index(snapshots: list) -> list:
    """
    Build a flat index of markable items from snapshots.

    Returns list of dicts with keys: id (e.g. "1", "1a"), key (for marked set),
    name (display), path, size, is_quantized.
    """
    index = []
    for i, snap in enumerate(snapshots, 1):
        # Full snapshot entry
        index.append({
            "id": str(i),
            "key": snap["name"],
            "name": snap["name"],
            "path": snap["path"],
            "size": snap["size"],
            "is_quantized": False,
        })
        # Quantized subfolder entries
        for j, quant in enumerate(snap.get("quantized", [])):
            letter = chr(ord('a') + j)
            index.append({
                "id": f"{i}{letter}",
                "key": f"{snap['name']}/{quant['name']}",
                "name": f"  └─ {quant['name']}",
                "path": quant["path"],
                "size": quant["size"],
                "is_quantized": True,
            })
    return index


def list_snapshots(run_info: dict, snapshots: list, marked: set) -> list:
    """Display snapshots for a run. Returns the index for command processing."""
    print(f"\nSnapshots for '{run_info['name']}' ({run_info['type']}):")
    print("-" * 65)

    if not snapshots:
        print("  No snapshots found.")
        return []

    index = build_snapshot_index(snapshots)

    for item in index:
        marked_str = "[X]" if item["key"] in marked else "[ ]"
        print(f"  {item['id']:<5} {item['name']:<25} {format_size(item['size']):<10} {marked_str}")

    print("-" * 65)
    if marked:
        marked_size = sum(item["size"] for item in index if item["key"] in marked)
        print(f"Marked: {len(marked)} items, {format_size(marked_size)}")
    print()

    return index


def snapshot_session(run_info: dict) -> set:
    """Interactive session to mark snapshots for deletion."""
    snapshots = get_snapshots(run_info)
    if not snapshots:
        print(f"No snapshots found for '{run_info['name']}'")
        return set()

    marked = set()
    index = build_snapshot_index(snapshots)
    id_to_item = {item["id"]: item for item in index}

    print("\nSnapshot commands:")
    print("  <id>      - Toggle mark (e.g., 1, 2, 1a, 2b)")
    print("  a         - Mark all")
    print("  n         - Unmark all")
    print("  l         - List snapshots")
    print("  d         - Done, delete marked items")
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
            marked = set(item["key"] for item in index)
            list_snapshots(run_info, snapshots, marked)
        elif cmd == "n":
            marked = set()
            list_snapshots(run_info, snapshots, marked)
        elif cmd in id_to_item:
            item = id_to_item[cmd]
            if item["key"] in marked:
                marked.remove(item["key"])
                print(f"Unmarked: {item['key']}")
            else:
                marked.add(item["key"])
                print(f"Marked: {item['key']}")
        else:
            valid_ids = ", ".join(sorted(id_to_item.keys(), key=lambda x: (len(x), x)))
            print(f"Unknown command. Valid IDs: {valid_ids}")

    return marked


def delete_snapshots(run_info: dict, snapshots: list, to_delete: set) -> None:
    """Delete marked snapshots and/or quantized subfolders."""
    for snap in snapshots:
        # Check if full snapshot is marked
        if snap["name"] in to_delete:
            print(f"  Deleting {snap['name']}...", end=" ")
            try:
                shutil.rmtree(snap["path"])
                print("done")
            except Exception as e:
                print(f"error: {e}")
        else:
            # Check for quantized subfolders
            for quant in snap.get("quantized", []):
                key = f"{snap['name']}/{quant['name']}"
                if key in to_delete:
                    print(f"  Deleting {key}...", end=" ")
                    try:
                        shutil.rmtree(quant["path"])
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

    sorted_keys = list_runs(runs, marked, snapshot_deletions)

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
            sorted_keys = list_runs(runs, marked, snapshot_deletions)
        elif cmd == "a":
            marked = set(runs.keys())
            list_runs(runs, marked, snapshot_deletions)
        elif cmd == "n":
            marked = set()
            list_runs(runs, marked, snapshot_deletions)
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
                    list_runs(runs, marked, snapshot_deletions)
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
            if info.get("rollouts") and info["rollouts"].exists():
                shutil.rmtree(info["rollouts"])
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
            if info.get("rollouts"):
                print(f"      {info['rollouts']}")

    # Show snapshots to delete
    if snapshot_deletions:
        print("\nSNAPSHOTS:")
        for key, marked_keys in snapshot_deletions.items():
            info = runs[key]
            snapshots = get_snapshots(info)
            index = build_snapshot_index(snapshots)
            snap_size = sum(item["size"] for item in index if item["key"] in marked_keys)
            total_size += snap_size
            print(f"  [{info['type']}] {info['name']}:")
            for item in index:
                if item["key"] in marked_keys:
                    print(f"      {item['key']} ({format_size(item['size'])})")

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
