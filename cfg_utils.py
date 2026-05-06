import re
import sys
from pathlib import Path
from typing import Dict, List


def _bootstrap_paths() -> None:
    """Make direct utility execution able to find the local maxlab package."""
    this_dir = Path(__file__).resolve().parent
    candidate_paths = [this_dir, this_dir.parent]

    maxlab_root = None
    for parent in this_dir.parents:
        if parent.name == "MaxLab":
            maxlab_root = parent
            break

    if maxlab_root is not None:
        lib_dir = maxlab_root / "python" / "lib"
        if lib_dir.exists():
            candidate_paths.extend(sorted(lib_dir.glob("python*/site-packages")))

    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _is_error_response(value) -> bool:
    """Return True for Maxwell API error-like responses."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "error"}
    return False


def parse_cfg(cfg_path: str) -> List[Dict]:
    """Parse a Maxwell GUI-exported cfg file into channel/electrode/position records."""
    with open(cfg_path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()

    pattern = re.compile(r"(\d+)\((\d+)\)([\d.]+)/([\d.]+)")
    entries: List[Dict] = []
    for match in pattern.finditer(raw):
        entries.append(
            {
                "channel": int(match.group(1)),
                "electrode": int(match.group(2)),
                "x": float(match.group(3)),
                "y": float(match.group(4)),
            }
        )
    return entries


def extract_electrodes(cfg_path: str) -> List[int]:
    """Extract unique electrodes from a Maxwell cfg file while preserving order."""
    seen = set()
    electrodes: List[int] = []
    for entry in parse_cfg(cfg_path):
        electrode = entry["electrode"]
        if electrode in seen:
            continue
        seen.add(electrode)
        electrodes.append(electrode)
    return electrodes


def merge_cfg_recording_electrodes(input_cfgs: List[str]) -> List[int]:
    """Merge recording electrodes from cfg files while preserving first-seen order."""
    if not input_cfgs:
        raise ValueError("input_cfgs cannot be empty.")

    seen = set()
    merged_electrodes: List[int] = []
    for cfg_path in input_cfgs:
        for electrode in extract_electrodes(cfg_path):
            if electrode in seen:
                continue
            seen.add(electrode)
            merged_electrodes.append(electrode)

    if not merged_electrodes:
        raise ValueError("No recording electrodes were found in input_cfgs.")
    return merged_electrodes


def export_recording_electrodes_to_cfg(
    recording_electrodes: List[int],
    output_cfg: str,
    array_name: str = "stimulation",
    electrode_weight: int = 1,
) -> str:
    """Route recording electrodes and export the array configuration as a .cfg file.

    This helper creates a Maxwell Array object, selects the given recording
    electrodes, runs routing, then saves the software-side array configuration
    to disk through Array.save_config(). It does not call Array.download().

    Parameters
    ----------
    recording_electrodes:
        Recording electrode IDs to route and export.
    output_cfg:
        Target .cfg file path. Parent directories are created when needed.
    array_name:
        Maxwell Array token name. Keep "stimulation" to match the experiment
        scripts unless there is a specific reason to use another token.
    electrode_weight:
        Routing priority passed to Array.select_electrodes().

    Returns
    -------
    str
        Absolute path of the generated .cfg file.
    """
    if not recording_electrodes:
        raise ValueError("recording_electrodes cannot be empty.")

    output_path = Path(output_cfg)
    if output_path.suffix.lower() != ".cfg":
        output_path = output_path.with_suffix(".cfg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _bootstrap_paths()
    import maxlab as mx

    array = mx.Array(array_name)
    reset_result = array.reset()
    if _is_error_response(reset_result):
        raise RuntimeError(f"Array.reset() failed: {reset_result}")

    clear_result = array.clear_selected_electrodes()
    if _is_error_response(clear_result):
        raise RuntimeError(f"Array.clear_selected_electrodes() failed: {clear_result}")

    select_result = array.select_electrodes(recording_electrodes, electrode_weight)
    if _is_error_response(select_result):
        raise RuntimeError(f"Array.select_electrodes() failed: {select_result}")

    route_result = array.route()
    if _is_error_response(route_result):
        raise RuntimeError(f"Array.route() failed: {route_result}")

    save_result = array.save_config(str(output_path))
    if save_result != 0:
        raise RuntimeError(f"Array.save_config() failed: {save_result}")

    return str(output_path.resolve())


def merge_cfg_recording_electrodes_to_cfg(
    input_cfgs: List[str],
    output_cfg: str,
    array_name: str = "stimulation",
    electrode_weight: int = 1,
) -> str:
    """Merge cfg recording electrodes, route them, and export a new cfg file."""
    merged_electrodes = merge_cfg_recording_electrodes(input_cfgs)
    return export_recording_electrodes_to_cfg(
        recording_electrodes=merged_electrodes,
        output_cfg=output_cfg,
        array_name=array_name,
        electrode_weight=electrode_weight,
    )


def _manual_export_recording_electrodes_cfg() -> None:
    """Manual entry for exporting selected recording electrodes to a cfg file."""
    recording_electrodes = [
        # Example:
        # 4896, 4897, 4898, 4899,
    ]
    output_cfg = "/home/maxwell/configs/selected_recording.cfg"
    array_name = "stimulation"
    electrode_weight = 1

    output_path = export_recording_electrodes_to_cfg(
        recording_electrodes=recording_electrodes,
        output_cfg=output_cfg,
        array_name=array_name,
        electrode_weight=electrode_weight,
    )
    print(
        f"[CFG] exported {len(recording_electrodes)} "
        f"recording electrodes to: {output_path}"
    )


def _manual_merge_cfgs() -> None:
    """Manual entry for merging cfg files into one routed cfg file."""
    input_cfgs = [
        # Example:
        # "/home/maxwell/configs/recording_a.cfg",
        # "/home/maxwell/configs/recording_b.cfg",
    ]
    output_cfg = "/home/maxwell/configs/merged_recording.cfg"
    array_name = "stimulation"
    electrode_weight = 1

    merged_electrodes = merge_cfg_recording_electrodes(input_cfgs)
    output_path = export_recording_electrodes_to_cfg(
        recording_electrodes=merged_electrodes,
        output_cfg=output_cfg,
        array_name=array_name,
        electrode_weight=electrode_weight,
    )
    print(
        f"[CFG] merged {len(input_cfgs)} cfg files into "
        f"{len(merged_electrodes)} recording electrodes: {output_path}"
    )


def main() -> None:
    """Manual runner for cfg utility functions.

    Keep this file as a cfg utility collection. When direct execution is needed,
    edit this function to call the specific helper for the current task.
    """
    _manual_export_recording_electrodes_cfg()
    # _manual_merge_cfgs()


if __name__ == "__main__":
    main()
