import getpass
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    from .cfg_utils import extract_electrodes as extract_cfg_electrodes
    from .recording_api import (
        preview_stim_mapping,
        setup_routing_and_units,
        start_recording,
        stop_recording,
        switch_stim_pool,
    )
    from .stimulation_api import (
        build_sequence_from_cfg,
        build_single_pulse_sequence,
        connect_units_subset,
        disconnect_all_units,
        merge_mapping_diagnostics,
        stimulate_units_random_order,
    )
    from .system_api import configure_and_powerup_stim_units, initialize_system
except ImportError:
    from cfg_utils import extract_electrodes as extract_cfg_electrodes
    from recording_api import (
        preview_stim_mapping,
        setup_routing_and_units,
        start_recording,
        stop_recording,
        switch_stim_pool,
    )
    from stimulation_api import (
        build_sequence_from_cfg,
        build_single_pulse_sequence,
        connect_units_subset,
        disconnect_all_units,
        merge_mapping_diagnostics,
        stimulate_units_random_order,
    )
    from system_api import configure_and_powerup_stim_units, initialize_system


def run_test_block(
    cfg: dict,
    stim_units_all: list[int],
    el2unit: dict[int, int],
    modes: list[dict] | None = None,
    block_label: str = "TEST",
) -> list[list[str]]:
    """Run one randomized test block across the selected modes."""
    test_cfg = cfg["test_block"]
    repeats = test_cfg["repeats"]
    gap_s = test_cfg["sleep_between_modes_s"]
    selected_modes = modes if modes is not None else test_cfg["modes"]

    all_orders: list[list[str]] = []

    for repeat_index in range(repeats):
        shuffled_modes = selected_modes.copy()
        import random

        random.shuffle(shuffled_modes)

        order_labels = [mode["name"] for mode in shuffled_modes]
        all_orders.append(order_labels)

        for mode in shuffled_modes:
            units_subset = [el2unit[electrode] for electrode in mode["electrodes"]]
            connect_units_subset(stim_units_all, units_subset)

            seq = build_single_pulse_sequence(
                cfg,
                label=f"{block_label}_repeat{repeat_index + 1}_{mode['name']}",
                pulse_config_key="test_pulse",
            )
            print(
                f"[{block_label}] repeat {repeat_index + 1} stimulate {mode['name']}"
                f" electrode_nums={len(mode['electrodes'])}"
            )
            seq.send()
            time.sleep(gap_s)

    disconnect_all_units(stim_units_all)
    return all_orders


def _sync_protocol_stim_electrodes_from_modes(cfg: dict) -> None:
    """Use the configured train/test modes as the protocol stimulation pool."""
    electrodes: list[int] = []
    seen: set[int] = set()

    def add_many(values: list[int]) -> None:
        for electrode in values:
            if electrode in seen:
                continue
            electrodes.append(electrode)
            seen.add(electrode)

    for mode in cfg["test_block"]["modes"]:
        add_many(mode["electrodes"])

    for pattern_name in cfg["protocol_flow"]["pattern_order"]:
        add_many(cfg["stim_patterns"][pattern_name]["mode10"])

    if not electrodes:
        raise ValueError("Protocol modes did not yield any stimulation electrodes.")

    cfg["stim_electrodes"] = electrodes
    print(
        f"[MAPPING] protocol stimulation pool built from modes: "
        f"{len(electrodes)} electrodes"
    )


def _modes_for_pattern(cfg: dict, pattern_name: str) -> list[dict]:
    """Return the configured test modes for one pattern."""
    modes = [
        mode for mode in cfg["test_block"]["modes"] if mode.get("pattern") == pattern_name
    ]
    if not modes:
        raise ValueError(f"No test modes configured for pattern: {pattern_name}")
    return modes


def run_train_block(
    cfg: dict,
    stim_units_all: list[int],
    el2unit: dict[int, int],
    pattern_name: str,
) -> None:
    """Run one train block with one pattern's complete 10mode."""
    train_cfg = cfg["train_block"]
    pulses = train_cfg["pulses"]
    freq_hz = train_cfg["freq_hz"]
    isi_s = 1.0 / freq_hz

    mode10 = cfg["stim_patterns"][pattern_name]["mode10"]
    units_subset = [el2unit[electrode] for electrode in mode10]
    connect_units_subset(stim_units_all, units_subset)

    for pulse_index in range(pulses):
        print(f"[TRAIN] {pattern_name} pulse {pulse_index + 1}/{pulses}")
        seq = build_single_pulse_sequence(
            cfg,
            label=f"TRAIN_{pattern_name}_pulse{pulse_index + 1}",
            pulse_config_key="train_pulse",
        )
        seq.send()
        time.sleep(isi_s)

    disconnect_all_units(stim_units_all)


def _summarize_range(values):
    """Compact a continuous integer list into a range descriptor when possible."""
    if not isinstance(values, list) or not values:
        return values
    if not all(isinstance(value, int) for value in values):
        return values

    start = values[0]
    expected = list(range(start, start + len(values)))
    if values != expected:
        return values

    return {
        "type": "range",
        "start": start,
        "stop": start + len(values),
        "count": len(values),
    }


def _resolve_recording_electrodes_for_json(cfg: dict) -> tuple[str, list[int]]:
    """Return the actual recording electrodes used for routing."""
    config_file = cfg.get("config", "")
    if config_file:
        return "config", extract_cfg_electrodes(config_file)
    return "fallback_recording_electrodes", cfg.get("recording_electrodes", [])


def _build_config_snapshot_for_json(cfg: dict) -> dict:
    """Build a readable config snapshot without losing experiment replay context."""
    snapshot = deepcopy(cfg)

    source, recording_electrodes = _resolve_recording_electrodes_for_json(cfg)
    snapshot["recording_electrodes_source"] = source
    snapshot["recording_electrodes_count"] = len(recording_electrodes)
    snapshot["recording_electrodes"] = recording_electrodes

    saving_cfg = snapshot.get("saving", {})
    if "group_channels" in saving_cfg:
        saving_cfg["group_channels"] = _summarize_range(saving_cfg["group_channels"])

    return snapshot


def _dump_json_readable(obj, indent: int = 0) -> str:
    """Pretty-print dicts while keeping scalar lists on one line."""
    pad = " " * indent
    next_pad = " " * (indent + 2)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lines = ["{"]
        items = list(obj.items())
        for index, (key, value) in enumerate(items):
            comma = "," if index < len(items) - 1 else ""
            rendered_value = _dump_json_readable(value, indent + 2)
            lines.append(f"{next_pad}{json.dumps(str(key), ensure_ascii=False)}: {rendered_value}{comma}")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(obj, list):
        if all(
            value is None or isinstance(value, (str, int, float, bool))
            for value in obj
        ):
            return json.dumps(obj, ensure_ascii=False)
        if not obj:
            return "[]"
        lines = ["["]
        for index, value in enumerate(obj):
            comma = "," if index < len(obj) - 1 else ""
            rendered_value = _dump_json_readable(value, indent + 2)
            lines.append(f"{next_pad}{rendered_value}{comma}")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    return json.dumps(obj, ensure_ascii=False)


def save_experiment_json(
    cfg: dict,
    out_name_prefix: str,
    random_orders=None,
    protocol_results=None,
    extra_meta: dict | None = None,
    timestamp: str | None = None,
) -> str:
    """Save one JSON summary into cfg['saving']['dir_name'].""" 
    out_dir = Path(cfg["saving"]["dir_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{out_name_prefix}_{timestamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "config": _build_config_snapshot_for_json(cfg),
    }

    if random_orders is not None:
        payload["random_experiment"] = {
            "random_orders": random_orders,
        }

    if protocol_results is not None:
        payload["protocol_experiment"] = {
            "protocol_results": protocol_results,
        }

    if extra_meta:
        payload["meta"] = extra_meta

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(_dump_json_readable(payload))
        handle.write("\n")

    print(f"[SAVE] JSON written to: {str(out_path)}")
    return str(out_path)


def run_mapping_dry_run(cfg: dict) -> dict:
    """Print the planned stim mapping without recording or protocol execution."""
    initialize_system()
    mapping_diag = preview_stim_mapping(cfg)

    print("[DRY-RUN] planned mapping summary")
    print(
        f"[DRY-RUN] requested={len(mapping_diag.get('requested_electrodes', []))} "
        f"resolved={len(mapping_diag.get('resolved_electrodes', []))} "
        f"substitutions={mapping_diag.get('n_substitutions', 0)} "
        f"conflict_units={mapping_diag.get('n_conflict_units', 0)}"
    )

    requested_to_resolved = mapping_diag.get("requested_to_resolved", {})
    electrode2unit = mapping_diag.get("electrode2unit", {})
    resolved_electrodes = [
        requested_to_resolved.get(str(requested), requested)
        for requested in mapping_diag.get("requested_electrodes", [])
    ]
    stim_units = [
        electrode2unit.get(str(requested), "?")
        for requested in mapping_diag.get("requested_electrodes", [])
    ]

    print(f"[DRY-RUN] requested electrodes: {mapping_diag.get('requested_electrodes', [])}")
    print(f"[DRY-RUN] resolved electrodes: {resolved_electrodes}")
    print(f"[DRY-RUN] stimulation units: {stim_units}")
    print(f"[DRY-RUN] resolved electrode/unit pairs: {list(zip(resolved_electrodes, stim_units))}")

    for requested in mapping_diag.get("requested_electrodes", []):
        requested_key = str(requested)
        resolved = requested_to_resolved.get(requested_key, requested)
        unit = electrode2unit.get(requested_key, "?")
        if resolved != requested:
            print(
                f"[DRY-RUN] requested electrode {requested} -> "
                f"resolved electrode {resolved} -> stim_unit {unit}"
            )
        else:
            print(f"[DRY-RUN] electrode {requested} -> stim_unit {unit}")

    if mapping_diag.get("conflicts"):
        print("[DRY-RUN] remaining conflicts:")
        print(json.dumps(mapping_diag["conflicts"], ensure_ascii=False, indent=2))
    elif mapping_diag.get("n_substitutions", 0) > 0:
        print(
            "[DRY-RUN] conflicts were resolved by substitutions; continue to final "
            "direct route check before the formal experiment."
        )
    else:
        print(
            "[DRY-RUN] requested electrodes have no conflicts in this mapping stage; "
            "continue to final direct route check."
        )

    return mapping_diag


def run_random_stim_experiment(cfg: dict) -> list[list[int]]:
    """Run the random-order single-unit response collection experiment."""
    initialize_system()
    _, stim_units = setup_routing_and_units(cfg)
    saving = None
    all_orders: list[list[int]] = []

    try:
        saving = start_recording(cfg)
        seq = build_sequence_from_cfg(cfg)
        configure_and_powerup_stim_units(stim_units)

        stim_cfg = cfg["random_stim"]
        all_orders = stimulate_units_random_order(
            seq=seq,
            stim_units=stim_units,
            stim_electrodes=cfg["stim_electrodes"],
            repeats=stim_cfg["repeats"],
            sleep_between_units_s=stim_cfg["sleep_between_units_s"],
            cfg=cfg,
        )
    except KeyboardInterrupt:
        print("[INTERRUPT] random experiment interrupted; partial recording will be closed.")
    finally:
        if saving is not None:
            stop_recording(saving)

    return all_orders


def run_train_block_control(
    cfg: dict,
    stim_units_all: list[int],
    el2unit: dict[int, int],
    pattern_name: str,
    cycle_index: int,
) -> list[dict]:
    """对照组训练块：每次脉冲从 32 池随机抽 N 个 unit 同时刺激。

    频率 / 脉冲数沿用 cfg["train_block"]，单脉冲波形沿用 cfg["train_pulse"]。
    每次抽样的 unit / electrode 列表写入返回的日志列表（同时打印到终端）。
    """
    import random

    train_cfg = cfg["train_block"]
    pulses = train_cfg["pulses"]
    freq_hz = train_cfg["freq_hz"]
    isi_s = 1.0 / freq_hz

    control_cfg = cfg["control_train"]
    n_random = control_cfg["n_random_per_pulse"]

    if n_random > len(stim_units_all):
        raise ValueError(
            f"control_train.n_random_per_pulse={n_random} exceeds "
            f"control pool size {len(stim_units_all)}."
        )

    unit_to_electrode = {unit: electrode for electrode, unit in el2unit.items()}

    sampling_log: list[dict] = []

    for pulse_index in range(pulses):
        sampled_units = random.sample(stim_units_all, n_random)
        sampled_electrodes = [unit_to_electrode[unit] for unit in sampled_units]

        connect_units_subset(stim_units_all, sampled_units)

        seq = build_single_pulse_sequence(
            cfg,
            label=(
                f"CTRL_TRAIN_{pattern_name}_cycle{cycle_index}_pulse{pulse_index + 1}"
            ),
            pulse_config_key="train_pulse",
        )
        print(
            f"[TRAIN][CTRL] {pattern_name} cycle {cycle_index} "
            f"pulse {pulse_index + 1}/{pulses} "
            f"units={sampled_units} electrodes={sampled_electrodes}"
        )
        seq.send()

        sampling_log.append(
            {
                "pulse_index": pulse_index + 1,
                "sampled_units": sampled_units,
                "sampled_electrodes": sampled_electrodes,
            }
        )

        time.sleep(isi_s)

    disconnect_all_units(stim_units_all)
    return sampling_log


def run_protocol(cfg: dict) -> dict:
    """Run pre-spontaneous/test/pattern training/post-spontaneous protocol."""
    proto_cfg = cfg["protocol_flow"]
    _sync_protocol_stim_electrodes_from_modes(cfg)

    results = {
        "test_orders": [],
        "protocol_steps": [],
        "status": "running",
        "interrupted": False,
    }
    saving = None

    def do_rest(seconds: float, tag: str) -> None:
        print(f"[REST] {tag}: {seconds:.1f}s")
        results["protocol_steps"].append(
            {"type": "rest", "tag": tag, "seconds": seconds}
        )
        time.sleep(seconds)

    def do_spontaneous(seconds: float, tag: str) -> None:
        print(f"[SPONTANEOUS] {tag}: {seconds:.1f}s")
        results["protocol_steps"].append(
            {"type": "spontaneous", "tag": tag, "seconds": seconds}
        )
        time.sleep(seconds)

    try:
        initialize_system()
        _, stim_units = setup_routing_and_units(cfg)
        saving = start_recording(cfg)
        configure_and_powerup_stim_units(stim_units)

        stim_electrodes = cfg["stim_electrodes"]
        el2unit = {el: unit for el, unit in zip(stim_electrodes, stim_units)}
        cfg["stim_mapping_diagnostics"] = merge_mapping_diagnostics(
            cfg.get("stim_mapping_diagnostics"),
            el2unit,
        )
        # 任务 7：信任 prechecked 输入；只在确实出现冲突时才打额外诊断。
        if cfg["stim_mapping_diagnostics"]["n_conflict_units"]:
            print(
                "[MAPPING]",
                cfg["stim_mapping_diagnostics"]["n_conflict_units"],
                "conflict units; extra_electrodes_due_to_conflicts=",
                cfg["stim_mapping_diagnostics"]["extra_electrodes_due_to_conflicts"],
            )

        do_spontaneous(proto_cfg["pre_spontaneous_s"], tag="pre_protocol")

        print("[FLOW] GLOBAL TEST")
        results["protocol_steps"].append({"type": "test", "tag": "global"})
        test_orders = run_test_block(
            cfg,
            stim_units_all=stim_units,
            el2unit=el2unit,
            modes=cfg["test_block"]["modes"],
            block_label="GLOBAL_TEST",
        )
        results["test_orders"].append(
            {
                "tag": "global",
                "mode_order": test_orders,
            }
        )

        pattern_order = proto_cfg["pattern_order"]
        for pattern_index, pattern_name in enumerate(pattern_order):
            pattern_modes = _modes_for_pattern(cfg, pattern_name)
            for cycle_index in range(1, proto_cfg["cycles_per_pattern"] + 1):
                print(f"[FLOW] {pattern_name} TRAIN #{cycle_index}")
                results["protocol_steps"].append(
                    {"type": "train", "pattern": pattern_name, "cycle": cycle_index}
                )
                run_train_block(
                    cfg,
                    stim_units_all=stim_units,
                    el2unit=el2unit,
                    pattern_name=pattern_name,
                )

                do_rest(
                    proto_cfg["rest_after_train_s"],
                    tag=f"after_{pattern_name}_train{cycle_index}",
                )

                print(f"[FLOW] {pattern_name} TEST #{cycle_index}")
                results["protocol_steps"].append(
                    {"type": "test", "pattern": pattern_name, "cycle": cycle_index}
                )
                test_orders = run_test_block(
                    cfg,
                    stim_units_all=stim_units,
                    el2unit=el2unit,
                    modes=pattern_modes,
                    block_label=f"{pattern_name}_TEST{cycle_index}",
                )
                results["test_orders"].append(
                    {
                        "tag": f"{pattern_name}_cycle{cycle_index}",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                        "mode_order": test_orders,
                    }
                )

            if pattern_index < len(pattern_order) - 1:
                do_rest(
                    proto_cfg["rest_between_patterns_s"],
                    tag=f"between_{pattern_name}_and_{pattern_order[pattern_index + 1]}",
                )

        do_spontaneous(proto_cfg["post_spontaneous_s"], tag="post_protocol")

        results["status"] = "completed"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        results["interrupted"] = True
        print("[INTERRUPT] protocol interrupted; partial results will be saved.")
    finally:
        if saving is not None:
            stop_recording(saving)

    return results


def run_protocol_control(cfg: dict) -> dict:
    """对照组 protocol：流程与实验组一致，train block 从对照组 32 池随机抽 N unit。

    流程：
      pre_spontaneous
      → GLOBAL TEST                          （实验组 routing）
      → for pattern in 4 patterns:
          for cycle in 1..N:
              switch → 对照组 routing
              CTRL TRAIN block               （32 池随机 10 unit / pulse）
              rest                            （兼做切换稳定时间）
              switch → 实验组 routing
              TEST block                      （pattern 自己的 4 mode）
          rest_between_patterns
      → post_spontaneous

    切换 routing 不调 mx.offset 与 waitAfterDownload，硬件稳定时间由 rest 时长承担。
    """
    proto_cfg = cfg["protocol_flow"]

    experimental_pool = list(cfg["experimental_stim_electrodes"])
    control_pool = list(cfg["control_stim_electrodes"])
    if not experimental_pool:
        raise ValueError("experimental_stim_electrodes must not be empty.")
    if not control_pool:
        raise ValueError("control_stim_electrodes must not be empty.")

    _sync_protocol_stim_electrodes_from_modes(cfg)

    results = {
        "experiment_group": "control",
        "test_orders": [],
        "protocol_steps": [],
        "control_train_log": [],
        "status": "running",
        "interrupted": False,
    }
    saving = None

    def do_rest(seconds: float, tag: str) -> None:
        print(f"[REST] {tag}: {seconds:.1f}s")
        results["protocol_steps"].append(
            {"type": "rest", "tag": tag, "seconds": seconds}
        )
        time.sleep(seconds)

    def do_spontaneous(seconds: float, tag: str) -> None:
        print(f"[SPONTANEOUS] {tag}: {seconds:.1f}s")
        results["protocol_steps"].append(
            {"type": "spontaneous", "tag": tag, "seconds": seconds}
        )
        time.sleep(seconds)

    try:
        initialize_system()

        # 启动期与实验组路径完全一致：select(record) + select_stim(实验组 32) +
        # route + connect+query + download + offset + clear_events。
        # cfg["stim_electrodes"] 已由 _sync_protocol_stim_electrodes_from_modes 设为实验组 32 池。
        # 后续每次切换 train↔test 时 switch_stim_pool 走完整 select+route 路径
        # 把 record + 新 stim 一起 route 后 download，确保每次 download 都含 record。
        exp_stim_electrodes = cfg["stim_electrodes"]
        array, exp_stim_units = setup_routing_and_units(cfg)
        saving = start_recording(cfg)
        configure_and_powerup_stim_units(exp_stim_units)

        exp_el2unit = {
            electrode: unit
            for electrode, unit in zip(exp_stim_electrodes, exp_stim_units)
        }
        cfg["stim_mapping_diagnostics"] = merge_mapping_diagnostics(
            cfg.get("stim_mapping_diagnostics"),
            exp_el2unit,
        )
        diag_conflicts = cfg["stim_mapping_diagnostics"].get("n_conflict_units", 0)
        if diag_conflicts:
            print(f"[MAPPING][CTRL] {diag_conflicts} conflict units in experimental pool")

        do_spontaneous(proto_cfg["pre_spontaneous_s"], tag="pre_protocol")

        print("[FLOW][CTRL] GLOBAL TEST (experimental routing)")
        results["protocol_steps"].append({"type": "test", "tag": "global"})
        global_orders = run_test_block(
            cfg,
            stim_units_all=exp_stim_units,
            el2unit=exp_el2unit,
            modes=cfg["test_block"]["modes"],
            block_label="GLOBAL_TEST",
        )
        results["test_orders"].append(
            {
                "tag": "global",
                "mode_order": global_orders,
            }
        )

        pattern_order = proto_cfg["pattern_order"]
        for pattern_index, pattern_name in enumerate(pattern_order):
            pattern_modes = _modes_for_pattern(cfg, pattern_name)
            for cycle_index in range(1, proto_cfg["cycles_per_pattern"] + 1):
                # === 切到对照组 stim 池：clear_selected + select(record + control) + route +
                # connect(control) + query + download，不做 offset/wait/clear_events/reset ===
                ctrl_stim_units, ctrl_el2unit = switch_stim_pool(
                    cfg,
                    array=array,
                    new_stim_electrodes=control_pool,
                    label="control",
                )
                results["protocol_steps"].append(
                    {
                        "type": "switch_routing",
                        "to": "control",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                    }
                )

                print(f"[FLOW][CTRL] {pattern_name} TRAIN #{cycle_index}")
                results["protocol_steps"].append(
                    {
                        "type": "train",
                        "group": "control",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                    }
                )
                sampling_log = run_train_block_control(
                    cfg,
                    stim_units_all=ctrl_stim_units,
                    el2unit=ctrl_el2unit,
                    pattern_name=pattern_name,
                    cycle_index=cycle_index,
                )
                results["control_train_log"].append(
                    {
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                        "samples": sampling_log,
                    }
                )

                # 训练后 rest，同时让对照组→实验组切换的硬件稳定时间被吸收。
                do_rest(
                    proto_cfg["rest_after_train_s"],
                    tag=f"after_{pattern_name}_train{cycle_index}",
                )

                # === 切回实验组 stim 池进行 test：同样走完整 select(record + experimental) + route ===
                exp_stim_units, exp_el2unit = switch_stim_pool(
                    cfg,
                    array=array,
                    new_stim_electrodes=exp_stim_electrodes,
                    label="experimental",
                )
                results["protocol_steps"].append(
                    {
                        "type": "switch_routing",
                        "to": "experimental",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                    }
                )

                print(f"[FLOW][CTRL] {pattern_name} TEST #{cycle_index}")
                results["protocol_steps"].append(
                    {
                        "type": "test",
                        "group": "experimental",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                    }
                )
                test_orders = run_test_block(
                    cfg,
                    stim_units_all=exp_stim_units,
                    el2unit=exp_el2unit,
                    modes=pattern_modes,
                    block_label=f"{pattern_name}_TEST{cycle_index}",
                )
                results["test_orders"].append(
                    {
                        "tag": f"{pattern_name}_cycle{cycle_index}",
                        "pattern": pattern_name,
                        "cycle": cycle_index,
                        "mode_order": test_orders,
                    }
                )

            if pattern_index < len(pattern_order) - 1:
                do_rest(
                    proto_cfg["rest_between_patterns_s"],
                    tag=f"between_{pattern_name}_and_{pattern_order[pattern_index + 1]}",
                )

        do_spontaneous(proto_cfg["post_spontaneous_s"], tag="post_protocol")

        results["status"] = "completed"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        results["interrupted"] = True
        print("[INTERRUPT][CTRL] protocol interrupted; partial results will be saved.")
    finally:
        if saving is not None:
            stop_recording(saving)

    return results
