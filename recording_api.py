import time

import maxlab as mx

try:
    from .stimulation_api import build_mapping_diag_from_el2unit
    from .system_api import (
        connect_stim_units_with_neighbor_retry,
        configure_array,
        connect_stim_units_to_stim_electrodes,
        expand_stim_electrode_pool,
    )
except ImportError:
    from stimulation_api import build_mapping_diag_from_el2unit
    from system_api import (
        connect_stim_units_with_neighbor_retry,
        configure_array,
        connect_stim_units_to_stim_electrodes,
        expand_stim_electrode_pool,
    )


def _build_routing_stim_electrodes(cfg: dict) -> list[int]:
    """Expand routed stimulation electrodes when neighbor-retry needs candidate neighbors."""
    stim_electrodes = cfg["stim_electrodes"]
    mapping_cfg = cfg.get("stim_mapping", {})
    mapping_strategy = mapping_cfg.get("strategy", "keep_conflicts")

    if mapping_strategy != "neighbor_retry":
        return stim_electrodes.copy()

    return expand_stim_electrode_pool(
        stim_electrodes,
        mapping_cfg.get("neighbor_radius", 50),
    )


def _validate_stim_electrode_limit(cfg: dict, stim_electrodes: list[int]) -> None:
    max_electrodes = cfg.get("stim_mapping", {}).get("max_electrodes", 32)
    if len(stim_electrodes) > max_electrodes:
        raise ValueError(
            f"Requested {len(stim_electrodes)} stimulation electrodes, "
            f"but the configured limit is {max_electrodes}."
        )


def _raise_if_direct_mapping_has_conflicts(mapping_diag: dict) -> None:
    conflicts = mapping_diag.get("conflicts", [])
    if not conflicts:
        print("[MAPPING] direct mapping has no conflicts; continue experiment.")
        return

    print("[MAPPING] direct mapping still has conflicts:")
    for conflict in conflicts:
        print(
            f"[MAPPING] stim_unit {conflict['stim_unit']} conflict: "
            f"electrodes {conflict['electrodes']}"
        )
    raise RuntimeError(
        "Formal experiment received a stimulation electrode set with stim-unit conflicts. "
        "Run mapping_preflight.py first, then import its resolved_electrodes into the experiment config."
    )


def _resolve_stim_mapping(cfg: dict, array: mx.Array) -> tuple[list[int], dict]:
    """Resolve electrode->stim-unit mapping according to the configured strategy."""
    stim_electrodes = cfg["stim_electrodes"]
    mapping_cfg = cfg.get("stim_mapping", {})
    mapping_strategy = mapping_cfg.get("strategy", "keep_conflicts")

    print(f"[MAPPING] strategy={mapping_strategy}")

    if mapping_strategy == "neighbor_retry":
        stim_units, mapping_diag = connect_stim_units_with_neighbor_retry(
            stim_electrodes=stim_electrodes,
            array=array,
            max_search_radius=mapping_cfg.get("neighbor_radius", 50),
            max_verification_iterations=mapping_cfg.get("verification_max_iterations", 8),
        )
    else:
        stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
        requested_el2unit = {
            electrode: unit for electrode, unit in zip(stim_electrodes, stim_units)
        }
        mapping_diag = {
            "strategy": mapping_strategy,
            "requested_electrodes": stim_electrodes.copy(),
            "resolved_electrodes": stim_electrodes.copy(),
            "requested_to_resolved": {
                str(electrode): electrode for electrode in stim_electrodes
            },
            "substitutions": {},
            "n_substitutions": 0,
            "fallback_conflict_requested_electrodes": [],
        }
        mapping_diag.update(build_mapping_diag_from_el2unit(requested_el2unit))
        mapping_diag["resolved_electrode2unit"] = {
            str(electrode): int(unit) for electrode, unit in requested_el2unit.items()
        }
        mapping_diag["electrode2unit"] = {
            str(electrode): int(unit) for electrode, unit in requested_el2unit.items()
        }

    cfg["stim_mapping_diagnostics"] = mapping_diag
    return stim_units, mapping_diag


def preview_stim_mapping(cfg: dict) -> dict:
    """Plan the stimulation mapping without download, recording, or protocol execution."""
    wells = cfg["wells"]
    rec_electrodes = cfg["recording_electrodes"]
    _validate_stim_electrode_limit(cfg, cfg["stim_electrodes"])
    routing_stim_electrodes = _build_routing_stim_electrodes(cfg)
    config_file = cfg["config"]

    # 按官方 8 步序列：mx.activate(wells) 必须早于 array.reset / select / route
    mx.activate(wells)
    array = configure_array(
        rec_electrodes,
        routing_stim_electrodes,
        config_file=config_file,
    )
    _, mapping_diag = _resolve_stim_mapping(cfg, array)
    return mapping_diag


def setup_routing_and_units(cfg: dict) -> tuple[mx.Array, list[int]]:
    """Run routing, build stim-unit mapping, then download and offset."""
    wells = cfg["wells"]
    rec_electrodes = cfg["recording_electrodes"]
    _validate_stim_electrode_limit(cfg, cfg["stim_electrodes"])
    routing_stim_electrodes = _build_routing_stim_electrodes(cfg)
    config_file = cfg["config"]
    mapping_strategy = cfg.get("stim_mapping", {}).get("strategy", "keep_conflicts")

    if mapping_strategy == "neighbor_retry":
        raise RuntimeError(
            "neighbor_retry is preflight-only. Run mapping_preflight.py, then put the "
            "resolved_electrodes into experiment_config.py before starting the formal experiment."
        )
    else:
        # 按官方 8 步序列：mx.activate(wells) 必须早于 array.reset / select / route
        mx.activate(wells)
        array = configure_array(
            rec_electrodes,
            routing_stim_electrodes,
            config_file=config_file,
        )
        stim_units, mapping_diag = _resolve_stim_mapping(cfg, array)
        _raise_if_direct_mapping_has_conflicts(mapping_diag)

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(3)
    mx.clear_events()

    return array, stim_units


def start_recording(cfg: dict) -> mx.Saving:
    """Open the target file and start recording."""
    saving_cfg = cfg["saving"]

    saving = mx.Saving()
    saving.open_directory(saving_cfg["dir_name"])
    saving.start_file(saving_cfg["file_name"])
    saving.group_define(
        0,
        saving_cfg["group_name"],
        saving_cfg["group_channels"],
    )
    saving.start_recording()
    print("Start recording")
    return saving


def stop_recording(saving: mx.Saving) -> None:
    """Stop recording and close the saving handles."""
    print("Stop recording")
    saving.stop_recording()
    time.sleep(mx.Timing.waitAfterRecording)
    saving.stop_file()
    saving.group_delete_all()
