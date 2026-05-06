import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import maxlab as mx

try:
    from .cfg_utils import extract_electrodes as extract_cfg_electrodes
except ImportError:
    try:
        from cfg_utils import extract_electrodes as extract_cfg_electrodes
    except ImportError:
        extract_cfg_electrodes = None


def initialize_system() -> None:
    """Initialize system into a defined state."""
    mx.initialize()
    # mxwserver 实测返回 'OK' 全大写；硬编码 'Ok' 在某些版本会误判失败。
    # 同时容忍前后空白与 None。
    result = mx.send(mx.Core().enable_stimulation_power(True))
    if (result or "").strip().upper() != "OK":
        raise RuntimeError("The system didn't initialize correctly.")


def configure_array(
    electrodes: List[int],
    stim_electrodes: List[int],
    config_file: Optional[str] = None,
) -> mx.Array:
    """Configure the recording and stimulation electrodes."""
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()

    if config_file:
        routing_electrodes = _extract_recording_electrodes_from_config(config_file)
        array.select_electrodes(routing_electrodes)
        array.select_stimulation_electrodes(stim_electrodes)
        array.route()
        return array

    array.select_electrodes(electrodes)
    array.select_stimulation_electrodes(stim_electrodes)
    array.route()
    return array


def configure_array_dual_pool(
    electrodes: List[int],
    primary_stim_electrodes: List[int],
    secondary_stim_electrodes: List[int],
    config_file: Optional[str] = None,
) -> mx.Array:
    """对照组专用：把两组 stim 电极合并到一次 select_stimulation_electrodes 一起 route。

    主组 + 副组联合声明，让 routing 算法同时为两组电极建立 amplifier 路由
    （Maxwell 单 well stim 选择上限 1020，64 远低于上限）。run-time stim_unit
    的具体分配在 connect_electrode_to_stimulation 阶段动态完成；切换时无需
    再次调 route，只需 disconnect 旧 / connect 新 / download。
    """
    array = mx.Array("stimulation")
    array.reset()
    array.clear_selected_electrodes()

    seen: set[int] = set()
    combined_stim: List[int] = []
    for electrode in list(primary_stim_electrodes) + list(secondary_stim_electrodes):
        if electrode in seen:
            continue
        seen.add(electrode)
        combined_stim.append(electrode)

    if config_file:
        routing_electrodes = _extract_recording_electrodes_from_config(config_file)
        array.select_electrodes(routing_electrodes)
    else:
        array.select_electrodes(electrodes)

    array.select_stimulation_electrodes(combined_stim)
    array.route()
    return array


def load_config(config_file: str) -> mx.Array:
    """Load a previously created configuration."""
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

    array = mx.Array("stimulation")
    try:
        array.load_config(config_file)
    except Exception as exc:
        raise Exception(f"Error loading config file '{config_file}': {str(exc)}")
    return array


def _extract_recording_electrodes_from_config(config_file: str) -> List[int]:
    """Parse recording electrodes from a GUI-exported cfg file or raise."""
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_file}' not found.")

    if extract_cfg_electrodes is None:
        raise ImportError(
            "cfg_utils.extract_electrodes is unavailable, so config-based routing "
            "cannot be used."
        )

    try:
        parsed = extract_cfg_electrodes(config_file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse recording electrodes from config '{config_file}': {exc}"
        ) from exc

    if not parsed:
        raise RuntimeError(
            f"Config '{config_file}' did not yield any recording electrodes."
        )

    return parsed


def connect_stim_units_to_stim_electrodes(
    stim_electrodes: List[int], array: mx.Array
) -> List[int]:
    """Build the stim electrode -> stim unit mapping."""
    stim_units: List[int] = []
    for stim_el in stim_electrodes:
        if not _has_routed_amplifier(array, stim_el):
            raise RuntimeError(
                f"Electrode {stim_el} is not routed to an amplifier, so it cannot be connected "
                "to a stimulation unit."
            )

        connect_result = array.connect_electrode_to_stimulation(stim_el)
        if _is_error_response(connect_result):
            raise RuntimeError(
                f"Failed to connect electrode {stim_el} to stimulation during final mapping."
            )

        stim = array.query_stimulation_at_electrode(stim_el)
        if _is_error_response(stim) or len(stim) == 0:
            raise RuntimeError(
                f"No stimulation channel can connect to electrode: {str(stim_el)}"
            )

        stim_unit_int = int(stim)
        if stim_unit_int in stim_units:
            # 输出一下信息 哪个电极和哪个电极连接在了同一个刺激单元了 刺激单元是多少
            print(f"Warning: Electrode {stim_el} is connected to stimulation unit {stim_unit_int}, which is already connected to another electrode. This may lead to unintended simultaneous stimulation of these electrodes.")
            stim_units.append(stim_unit_int)
        else:
            stim_units.append(stim_unit_int)

    return stim_units


def _logical_distance_sq(electrode_a: int, electrode_b: int) -> int:
    """Compute squared grid distance on the MaxOne electrode lattice."""
    num_cols = 220
    row_a, col_a = divmod(electrode_a, num_cols)
    row_b, col_b = divmod(electrode_b, num_cols)
    return (row_a - row_b) ** 2 + (col_a - col_b) ** 2


def _is_error_response(response: Optional[str]) -> bool:
    """Return whether the low-level API reported an error string."""
    return response is None or str(response).strip().lower() == "error"


def _has_routed_amplifier(array: mx.Array, electrode: int) -> bool:
    """Check whether an electrode is currently routed to an amplifier."""
    amplifier = array.query_amplifier_at_electrode(electrode)
    return not _is_error_response(amplifier) and len(amplifier) > 0


def _build_candidate_pool(
    original_electrode: int,
    max_search_radius: int,
) -> List[Tuple[int, int, int]]:
    """Return ordered candidate electrodes as (electrode, radius, distance_sq)."""
    seen = {original_electrode}
    candidates: List[Tuple[int, int, int]] = [(original_electrode, 0, 0)]

    for radius in range(1, max_search_radius + 1):
        neighbors = mx.util.electrode_neighbors(original_electrode, radius)
        ordered_neighbors = sorted(
            (neighbor for neighbor in neighbors if neighbor not in seen),
            key=lambda neighbor: (
                _logical_distance_sq(original_electrode, neighbor),
                abs(neighbor - original_electrode),
                neighbor,
            ),
        )
        for neighbor in ordered_neighbors:
            seen.add(neighbor)
            candidates.append(
                (
                    neighbor,
                    radius,
                    _logical_distance_sq(original_electrode, neighbor),
                )
            )

    return candidates


def expand_stim_electrode_pool(
    stim_electrodes: List[int],
    max_search_radius: int,
) -> List[int]:
    """Expand the routed stimulation pool to cover neighbor-retry candidates."""
    expanded: List[int] = []
    seen: set[int] = set()

    for original_electrode in stim_electrodes:
        for candidate, _, _ in _build_candidate_pool(original_electrode, max_search_radius):
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)

    return expanded


def _probe_stim_unit(array: mx.Array, electrode: int) -> Optional[int]:
    """Probe which stimulation unit a candidate electrode maps to."""
    if not _has_routed_amplifier(array, electrode):
        return None

    connect_result = array.connect_electrode_to_stimulation(electrode)
    if _is_error_response(connect_result):
        return None

    stim = array.query_stimulation_at_electrode(electrode)

    disconnect_result = array.disconnect_electrode_from_stimulation(electrode)
    if _is_error_response(disconnect_result):
        return None

    if _is_error_response(stim) or len(stim) == 0:
        return None
    return int(stim)


def _build_conflict_summary(el2unit: Dict[int, int]) -> Dict:
    """Build JSON-safe conflict diagnostics for an electrode->unit mapping."""
    unit2els: Dict[int, List[int]] = {}
    for electrode, unit in el2unit.items():
        unit2els.setdefault(unit, []).append(electrode)

    for unit in unit2els:
        unit2els[unit] = sorted(unit2els[unit])

    conflicts = [
        {"stim_unit": unit, "electrodes": electrodes, "count": len(electrodes)}
        for unit, electrodes in unit2els.items()
        if len(electrodes) > 1
    ]
    conflicts = sorted(conflicts, key=lambda item: (-item["count"], item["stim_unit"]))

    return {
        "n_electrodes": len(el2unit),
        "n_units_unique": len(set(el2unit.values())),
        "n_conflict_units": len(conflicts),
        "extra_electrodes_due_to_conflicts": sum(
            conflict["count"] - 1 for conflict in conflicts
        ),
        "conflicts": conflicts,
        "unit2electrodes": {str(unit): electrodes for unit, electrodes in unit2els.items()},
        "electrode2unit": {str(electrode): int(unit) for electrode, unit in el2unit.items()},
    }


def _print_mapping_arrays(
    stim_electrodes: List[int],
    requested_to_resolved: Dict[str, int],
    requested_el2unit: Dict[int, int],
    prefix: str,
) -> None:
    """Print compact requested/resolved/unit arrays for the current mapping."""
    resolved_electrodes = [requested_to_resolved[str(electrode)] for electrode in stim_electrodes]
    stim_units = [requested_el2unit[electrode] for electrode in stim_electrodes]
    print(f"{prefix} requested electrodes: {stim_electrodes}")
    print(f"{prefix} resolved electrodes: {resolved_electrodes}")
    print(f"{prefix} stimulation units: {stim_units}")
    print(f"{prefix} resolved electrode/unit pairs: {list(zip(resolved_electrodes, stim_units))}")


def _print_conflict_details(
    summary: Dict,
    requested_to_resolved: Dict[str, int],
    prefix: str,
) -> None:
    """Print which requested electrodes currently collide on which stimulation units."""
    conflicts = summary.get("conflicts", [])
    if not conflicts:
        return

    for conflict in conflicts:
        unit = conflict["stim_unit"]
        requested = conflict["electrodes"]
        resolved_pairs = [
            f"{electrode}->{requested_to_resolved.get(str(electrode), electrode)}"
            for electrode in requested
        ]
        print(
            f"{prefix} stim_unit {unit} conflict: requested electrodes {requested}; "
            f"current resolved {resolved_pairs}"
        )


def _select_candidate_plan(
    stim_electrodes: List[int],
    candidate_records: Dict[int, List[Dict[str, int]]],
    banned_resolved_by_original: Dict[int, set[int]],
    original_order: Dict[int, int],
) -> Tuple[Dict[int, Dict[str, int]], List[int]]:
    """Choose one candidate per requested electrode with a greedy uniqueness bias."""

    def available_candidates(original_electrode: int) -> List[Dict[str, int]]:
        banned = banned_resolved_by_original.get(original_electrode, set())
        available = [
            candidate
            for candidate in candidate_records[original_electrode]
            if candidate["electrode"] not in banned
        ]
        return available or candidate_records[original_electrode]

    planning_order = sorted(
        stim_electrodes,
        key=lambda electrode: (
            len({candidate["stim_unit"] for candidate in available_candidates(electrode)}),
            len(available_candidates(electrode)),
            original_order[electrode],
        ),
    )

    selected_by_original: Dict[int, Dict[str, int]] = {}
    used_resolved_electrodes: set[int] = set()
    unit_usage_count: Dict[int, int] = {}
    fallback_conflict_requested: List[int] = []

    def assign_candidate(original_electrode: int, candidate: Dict[str, int]) -> None:
        selected_by_original[original_electrode] = candidate
        used_resolved_electrodes.add(candidate["electrode"])
        stim_unit = candidate["stim_unit"]
        unit_usage_count[stim_unit] = unit_usage_count.get(stim_unit, 0) + 1

    deferred: List[int] = []
    for original_electrode in planning_order:
        choice = None
        for candidate in available_candidates(original_electrode):
            if candidate["electrode"] in used_resolved_electrodes:
                continue
            if unit_usage_count.get(candidate["stim_unit"], 0) == 0:
                choice = candidate
                break

        if choice is None:
            deferred.append(original_electrode)
            continue

        assign_candidate(original_electrode, choice)

    for original_electrode in deferred:
        available = available_candidates(original_electrode)
        non_reused_electrode_candidates = [
            candidate
            for candidate in available
            if candidate["electrode"] not in used_resolved_electrodes
        ]
        fallback_pool = non_reused_electrode_candidates or available
        choice = min(
            fallback_pool,
            key=lambda candidate: (
                unit_usage_count.get(candidate["stim_unit"], 0),
                candidate["radius"],
                candidate["distance_sq"],
                abs(candidate["electrode"] - original_electrode),
                candidate["electrode"],
            ),
        )
        assign_candidate(original_electrode, choice)
        if unit_usage_count[choice["stim_unit"]] > 1:
            fallback_conflict_requested.append(original_electrode)

    return selected_by_original, fallback_conflict_requested


def _verify_selected_mapping(
    array: mx.Array,
    stim_electrodes: List[int],
    selected_by_original: Dict[int, Dict[str, int]],
) -> Dict:
    """Connect the full resolved set, then query actual simultaneous mapping."""
    connected_resolved_electrodes: List[int] = []
    requested_el2unit: Dict[int, int] = {}
    resolved_el2unit: Dict[int, int] = {}
    unresolved_requested: List[int] = []

    try:
        already_connected: set[int] = set()
        for original_electrode in stim_electrodes:
            resolved_electrode = selected_by_original[original_electrode]["electrode"]
            if resolved_electrode in already_connected:
                continue
            if not _has_routed_amplifier(array, resolved_electrode):
                unresolved_requested.append(original_electrode)
                continue

            connect_result = array.connect_electrode_to_stimulation(resolved_electrode)
            if _is_error_response(connect_result):
                unresolved_requested.append(original_electrode)
                continue

            connected_resolved_electrodes.append(resolved_electrode)
            already_connected.add(resolved_electrode)

        for original_electrode in stim_electrodes:
            if original_electrode in unresolved_requested:
                continue

            resolved_electrode = selected_by_original[original_electrode]["electrode"]
            stim = array.query_stimulation_at_electrode(resolved_electrode)

            if _is_error_response(stim) or len(stim) == 0:
                unresolved_requested.append(original_electrode)
                continue

            stim_unit = int(stim)
            requested_el2unit[original_electrode] = stim_unit
            resolved_el2unit[resolved_electrode] = stim_unit
    finally:
        for resolved_electrode in reversed(connected_resolved_electrodes):
            disconnect_result = array.disconnect_electrode_from_stimulation(
                resolved_electrode
            )
            if _is_error_response(disconnect_result):
                print(
                    f"[MAPPING] warning: failed to disconnect resolved electrode "
                    f"{resolved_electrode} during verification cleanup."
                )

    diag = _build_conflict_summary(requested_el2unit)
    conflict_units = {item["stim_unit"] for item in diag["conflicts"]}
    conflict_requested = sorted(
        original_electrode
        for original_electrode, stim_unit in requested_el2unit.items()
        if stim_unit in conflict_units
    )

    for original_electrode in unresolved_requested:
        if original_electrode not in conflict_requested:
            conflict_requested.append(original_electrode)

    conflict_requested.sort()

    return {
        "requested_el2unit": requested_el2unit,
        "resolved_el2unit": resolved_el2unit,
        "unresolved_requested_electrodes": sorted(set(unresolved_requested)),
        "conflict_requested_electrodes": conflict_requested,
        "summary": diag,
    }


def _connect_resolved_electrodes(array: mx.Array, resolved_electrodes: List[int]) -> None:
    """Leave the final resolved electrode set connected on the array object."""
    connected_once: set[int] = set()
    for resolved_electrode in resolved_electrodes:
        if resolved_electrode in connected_once:
            continue
        if not _has_routed_amplifier(array, resolved_electrode):
            raise RuntimeError(
                f"Resolved electrode {resolved_electrode} is not routed to an amplifier "
                "in the final array."
            )
        connect_result = array.connect_electrode_to_stimulation(resolved_electrode)
        if _is_error_response(connect_result):
            raise RuntimeError(
                f"Failed to connect resolved electrode {resolved_electrode} in the final array."
            )
        connected_once.add(resolved_electrode)


def connect_stim_units_with_neighbor_retry(
    stim_electrodes: List[int],
    array: mx.Array,
    max_search_radius: int = 50,
    max_verification_iterations: int = 100,
) -> Tuple[List[int], Dict]:
    """Resolve stim-unit conflicts with neighbor search plus full-set verification."""
    candidate_records: Dict[int, List[Dict[str, int]]] = {}
    original_order = {electrode: index for index, electrode in enumerate(stim_electrodes)}

    for original_electrode in stim_electrodes:
        candidates: List[Dict[str, int]] = []
        for candidate, radius, distance_sq in _build_candidate_pool(
            original_electrode,
            max_search_radius,
        ):
            stim_unit = _probe_stim_unit(array, candidate)
            if stim_unit is None:
                continue
            candidates.append(
                {
                    "electrode": candidate,
                    "stim_unit": stim_unit,
                    "radius": radius,
                    "distance_sq": distance_sq,
                }
            )
        candidate_records[original_electrode] = candidates

    unroutable = [
        electrode for electrode, candidates in candidate_records.items() if not candidates
    ]
    if unroutable:
        raise RuntimeError(
            "No routable stimulation candidate found within radius "
            f"{max_search_radius} for electrodes: {unroutable}"
        )

    direct_requested_el2unit: Dict[int, int] = {}
    direct_unroutable: List[int] = []
    direct_requested_to_resolved: Dict[str, int] = {}
    for original_electrode in stim_electrodes:
        direct_candidates = [
            candidate
            for candidate in candidate_records[original_electrode]
            if candidate["radius"] == 0
        ]
        if not direct_candidates:
            direct_unroutable.append(original_electrode)
            continue
        direct_requested_el2unit[original_electrode] = direct_candidates[0]["stim_unit"]
        direct_requested_to_resolved[str(original_electrode)] = original_electrode

    direct_summary = _build_conflict_summary(direct_requested_el2unit)
    if direct_unroutable:
        print(
            "[MAPPING] direct mapping unavailable for requested electrodes: "
            f"{direct_unroutable}"
        )
    if direct_summary["n_conflict_units"] == 0 and len(direct_requested_el2unit) == len(
        stim_electrodes
    ):
        print("[MAPPING] direct mapping has no conflicts.")
        _print_mapping_arrays(
            stim_electrodes,
            direct_requested_to_resolved,
            direct_requested_el2unit,
            prefix="[MAPPING][direct]",
        )
    else:
        print("[MAPPING] direct mapping has conflicts; start neighbor retry.")
        _print_mapping_arrays(
            stim_electrodes,
            direct_requested_to_resolved,
            direct_requested_el2unit,
            prefix="[MAPPING][direct]",
        )
        _print_conflict_details(
            direct_summary,
            direct_requested_to_resolved,
            prefix="[MAPPING][direct]",
        )

    banned_resolved_by_original: Dict[int, set[int]] = {}
    attempted_signatures: set[Tuple[int, ...]] = set()
    best_attempt: Optional[Dict] = None

    def build_attempt_payload(
        selected_by_original: Dict[int, Dict[str, int]],
        fallback_conflict_requested: List[int],
        verification: Dict,
        iteration_index: int,
    ) -> Dict:
        final_electrodes: List[int] = []
        requested_to_resolved: Dict[str, int] = {}
        substitutions: Dict[str, int] = {}

        for original_electrode in stim_electrodes:
            resolved_electrode = selected_by_original[original_electrode]["electrode"]
            final_electrodes.append(resolved_electrode)
            requested_to_resolved[str(original_electrode)] = resolved_electrode
            if resolved_electrode != original_electrode:
                substitutions[str(original_electrode)] = resolved_electrode

        unresolved_count = len(verification["unresolved_requested_electrodes"])
        summary = verification["summary"]
        score = (
            unresolved_count,
            summary["n_conflict_units"],
            summary["extra_electrodes_due_to_conflicts"],
            sum(selected_by_original[electrode]["radius"] for electrode in stim_electrodes),
            sum(
                selected_by_original[electrode]["distance_sq"]
                for electrode in stim_electrodes
            ),
            len(substitutions),
        )

        return {
            "iteration": iteration_index,
            "selected_by_original": selected_by_original,
            "fallback_conflict_requested": fallback_conflict_requested,
            "verification": verification,
            "final_electrodes": final_electrodes,
            "requested_to_resolved": requested_to_resolved,
            "substitutions": substitutions,
            "score": score,
        }

    for iteration_index in range(1, max_verification_iterations + 1):
        selected_by_original, fallback_conflict_requested = _select_candidate_plan(
            stim_electrodes,
            candidate_records,
            banned_resolved_by_original,
            original_order,
        )
        signature = tuple(
            selected_by_original[original_electrode]["electrode"]
            for original_electrode in stim_electrodes
        )
        if signature in attempted_signatures:
            break
        attempted_signatures.add(signature)

        verification = _verify_selected_mapping(array, stim_electrodes, selected_by_original)
        attempt = build_attempt_payload(
            selected_by_original,
            fallback_conflict_requested,
            verification,
            iteration_index,
        )

        print(
            f"[MAPPING][iter {iteration_index}] verification: "
            f"conflict_units={verification['summary']['n_conflict_units']}, "
            f"unresolved={len(verification['unresolved_requested_electrodes'])}, "
            f"substitutions={len(attempt['substitutions'])}"
        )
        _print_conflict_details(
            verification["summary"],
            attempt["requested_to_resolved"],
            prefix=f"[MAPPING][iter {iteration_index}]",
        )
        if verification["unresolved_requested_electrodes"]:
            print(
                f"[MAPPING][iter {iteration_index}] unresolved requested electrodes: "
                f"{verification['unresolved_requested_electrodes']}"
            )

        if best_attempt is None or attempt["score"] < best_attempt["score"]:
            best_attempt = attempt

        if (
            not verification["conflict_requested_electrodes"]
            and not verification["unresolved_requested_electrodes"]
        ):
            best_attempt = attempt
            print(f"[MAPPING][iter {iteration_index}] full-set verification is now conflict-free.")
            _print_mapping_arrays(
                stim_electrodes,
                attempt["requested_to_resolved"],
                verification["requested_el2unit"],
                prefix=f"[MAPPING][iter {iteration_index}]",
            )
            break

        progress = False
        for original_electrode in verification["conflict_requested_electrodes"]:
            current_resolved = selected_by_original[original_electrode]["electrode"]
            current_unit = verification["requested_el2unit"].get(original_electrode, "?")
            alternatives = [
                candidate
                for candidate in candidate_records[original_electrode]
                if candidate["electrode"] != current_resolved
                and candidate["electrode"]
                not in banned_resolved_by_original.get(original_electrode, set())
            ]
            if not alternatives:
                print(
                    f"[MAPPING][iter {iteration_index}] requested electrode "
                    f"{original_electrode} currently conflicts at "
                    f"{current_resolved}->{current_unit}, but no new candidate remains."
                )
                continue
            next_candidate = alternatives[0]
            print(
                f"[MAPPING][iter {iteration_index}] requested electrode "
                f"{original_electrode} currently conflicts at "
                f"{current_resolved}->{current_unit}; next try "
                f"{next_candidate['electrode']}->predicted stim_unit "
                f"{next_candidate['stim_unit']}"
            )
            banned_resolved_by_original.setdefault(original_electrode, set()).add(
                current_resolved
            )
            progress = True

        if not progress:
            break

    if best_attempt is None:
        raise RuntimeError("neighbor_retry failed to produce any stimulation mapping plan.")

    final_electrodes = best_attempt["final_electrodes"]
    requested_to_resolved = best_attempt["requested_to_resolved"]
    substitutions = best_attempt["substitutions"]
    verification = best_attempt["verification"]
    resolved_el2unit = verification["resolved_el2unit"]
    requested_el2unit = verification["requested_el2unit"]
    summary = verification["summary"]
    unresolved_requested = verification["unresolved_requested_electrodes"]
    verified_conflict_requested = verification["conflict_requested_electrodes"]

    _connect_resolved_electrodes(array, final_electrodes)

    for original_electrode, resolved_electrode in substitutions.items():
        print(
            f"[FALLBACK] requested electrode {original_electrode} -> "
            f"resolved electrode {resolved_electrode}"
        )
    if verified_conflict_requested:
        print(
            "[MAPPING] full-set verification still reports conflicts for requested electrodes "
            f"{verified_conflict_requested}"
        )
    if unresolved_requested:
        print(
            "[MAPPING] full-set verification could not query requested electrodes "
            f"{unresolved_requested}"
        )
    if not verified_conflict_requested and not unresolved_requested:
        print("[MAPPING] final full-set verification has no conflicts.")
        _print_mapping_arrays(
            stim_electrodes,
            requested_to_resolved,
            requested_el2unit,
            prefix="[MAPPING][final]",
        )
    else:
        print("[MAPPING] final mapping still has unresolved items; best-attempt summary below.")
        _print_mapping_arrays(
            stim_electrodes,
            requested_to_resolved,
            requested_el2unit,
            prefix="[MAPPING][final]",
        )
        _print_conflict_details(
            summary,
            requested_to_resolved,
            prefix="[MAPPING][final]",
        )

    diag = dict(summary)
    diag.update(
        {
            "strategy": "neighbor_retry",
            "neighbor_radius": max_search_radius,
            "requested_electrodes": stim_electrodes.copy(),
            "resolved_electrodes": final_electrodes.copy(),
            "requested_to_resolved": requested_to_resolved,
            "resolved_electrode2unit": {
                str(electrode): int(unit) for electrode, unit in resolved_el2unit.items()
            },
            "substitutions": substitutions,
            "n_substitutions": len(substitutions),
            "fallback_conflict_requested_electrodes": best_attempt[
                "fallback_conflict_requested"
            ],
            "verified_conflict_requested_electrodes": verified_conflict_requested,
            "unresolved_requested_electrodes": unresolved_requested,
            "verification_iterations": best_attempt["iteration"],
        }
    )

    stim_units = [requested_el2unit.get(original_electrode) for original_electrode in stim_electrodes]
    if any(stim_unit is None for stim_unit in stim_units):
        raise RuntimeError(
            "Failed to verify a stimulation unit for all requested electrodes after "
            f"{best_attempt['iteration']} neighbor-retry iterations."
        )

    return [int(stim_unit) for stim_unit in stim_units], diag


def powerup_stim_unit(stim_unit: int) -> mx.StimulationUnit:
    """Power up and connect a specific stimulation unit."""
    return (
        mx.StimulationUnit(stim_unit)
        .power_up(True)
        .connect(True)
        .set_voltage_mode()
        .dac_source(0)
    )


def configure_and_powerup_stim_units(
    stim_units: List[int],
) -> List[mx.StimulationUnit]:
    """Configure and power up all stimulation units."""
    stim_unit_commands: List[mx.StimulationUnit] = []
    for stim_unit in stim_units:
        stim = powerup_stim_unit(stim_unit)
        stim_unit_commands.append(stim)
        mx.send(stim)
    return stim_unit_commands


def poweroff_all_stim_units() -> None:
    """Power off all stimulation units."""
    for stimulation_unit in range(0, 32):
        stim = mx.StimulationUnit(stimulation_unit)
        stim.power_up(False)
        stim.connect(False)
        mx.send(stim)
