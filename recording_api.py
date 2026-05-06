import time

import maxlab as mx

try:
    from .cfg_utils import extract_electrodes as extract_cfg_electrodes
    from .stimulation_api import build_mapping_diag_from_el2unit
    from .system_api import (
        configure_and_powerup_stim_units,
        configure_array,
        configure_array_dual_pool,
        connect_stim_units_to_stim_electrodes,
        connect_stim_units_with_neighbor_retry,
        expand_stim_electrode_pool,
        poweroff_all_stim_units,
    )
except ImportError:
    from cfg_utils import extract_electrodes as extract_cfg_electrodes
    from stimulation_api import build_mapping_diag_from_el2unit
    from system_api import (
        configure_and_powerup_stim_units,
        configure_array,
        configure_array_dual_pool,
        connect_stim_units_to_stim_electrodes,
        connect_stim_units_with_neighbor_retry,
        expand_stim_electrode_pool,
        poweroff_all_stim_units,
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
    """Trust prechecked input: if no conflicts, output one line and continue.

    用户预检过的电极组本就应该无冲突；正式实验信任输入，发现冲突立即报错让用户回到 mapping_preflight。
    无冲突时只输出一行简洁日志，不重复打印池子内容。
    """
    conflicts = mapping_diag.get("conflicts", [])
    n_electrodes = mapping_diag.get("n_electrodes", 0)
    if not conflicts:
        print(
            f"[MAPPING] {n_electrodes} prechecked electrodes -> "
            f"{mapping_diag.get('n_units_unique', 0)} stim_units, no conflicts; continue."
        )
        return

    print(f"[MAPPING] prechecked input has {len(conflicts)} stim_unit conflicts:")
    for conflict in conflicts:
        print(
            f"[MAPPING]   stim_unit {conflict['stim_unit']} <- electrodes {conflict['electrodes']}"
        )
    raise RuntimeError(
        "Formal experiment received a stimulation electrode set with stim-unit conflicts. "
        "Run mapping_preflight.py first, then import its resolved_electrodes into the experiment config."
    )


def _resolve_stim_mapping(cfg: dict, array: mx.Array) -> tuple[list[int], dict]:
    """Resolve electrode->stim-unit mapping according to the configured strategy.

    正式实验默认 strategy=prechecked → 直接 connect+query，无冲突静默通过；
    neighbor_retry 仅 mapping_preflight 用。日志在无冲突时只输出一行确认，避免冗余。
    """
    stim_electrodes = cfg["stim_electrodes"]
    mapping_cfg = cfg.get("stim_mapping", {})
    mapping_strategy = mapping_cfg.get("strategy", "keep_conflicts")

    if mapping_strategy == "neighbor_retry":
        print(f"[MAPPING] strategy={mapping_strategy} (preflight only)")
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
    """正式实验启动：select_record + select_stim + route + connect+query + download + offset。

    正式实验严格走 ``strategy=prechecked`` 直连：信任 mapping_preflight 已交付的无冲突电极组，
    不在此处再跑搜索；只 connect+query 拿 unit_id 后立即报告无冲突 / 报错。
    任何 ``neighbor_retry`` 都被 main.py 与本函数双重拦截，避免正式实验误走预检策略。
    """
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

    print(
        f"[MAPPING] formal run -> direct prechecked mapping "
        f"({len(cfg['stim_electrodes'])} stim electrodes, no search)"
    )

    # 按官方 8 步序列：mx.activate(wells) 必须早于 array.reset / select / route
    mx.activate(wells)
    array = configure_array(
        rec_electrodes,
        routing_stim_electrodes,
        config_file=config_file,
    )
    stim_units, mapping_diag = _resolve_stim_mapping(cfg, array)
    _raise_if_direct_mapping_has_conflicts(mapping_diag)

    # 印一行 electrode->unit 对照，便于与 mapping_preflight 输出做交叉校验。
    pairs = list(zip(cfg["stim_electrodes"], stim_units))
    print(f"[MAPPING] electrode->unit pairs: {pairs}")

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(3)
    mx.clear_events()

    return array, stim_units


def setup_routing_dual_stim_pools(
    cfg: dict,
    primary_pool: list[int],
    secondary_pool: list[int],
) -> tuple[mx.Array, list[int], dict[int, int]]:
    """[DEPRECATED] 双池合并启动方案。

    放弃原因：用户 2026-05-06 实测肯定确认 —— 即使两组 stim 电极都在启动期 select,
    切换 stim 桥接 + download 仍**无法**让新 stim 在硬件层生效。每次 download 之前必须
    重新 select_electrodes(record) + route。当前 ``run_protocol_control`` 启动期改回
    ``setup_routing_and_units``（与实验组同路径），切换期 ``switch_stim_pool`` 走完整
    select(record + new_stim) + route + download。

    保留本函数仅供历史对照与未来再次评估，**新代码不要调用**。
    """
    wells = cfg["wells"]
    rec_electrodes = cfg["recording_electrodes"]
    config_file = cfg["config"]
    mapping_strategy = cfg.get("stim_mapping", {}).get("strategy", "keep_conflicts")

    if mapping_strategy == "neighbor_retry":
        raise RuntimeError(
            "setup_routing_dual_stim_pools requires prechecked electrodes for both pools. "
            "neighbor_retry is preflight-only."
        )
    if not primary_pool:
        raise ValueError("primary_pool must not be empty.")
    if not secondary_pool:
        raise ValueError("secondary_pool must not be empty.")

    _validate_stim_electrode_limit(cfg, primary_pool)
    _validate_stim_electrode_limit(cfg, secondary_pool)

    print(
        f"[SETUP][DUAL] joint routing: primary={len(primary_pool)} + "
        f"secondary={len(secondary_pool)} stim electrodes"
    )

    # 按官方启动序列：mx.activate(wells) 必须早于 array.reset / select / route。
    mx.activate(wells)
    array = configure_array_dual_pool(
        rec_electrodes,
        primary_pool,
        secondary_pool,
        config_file=config_file,
    )

    # 启动期只对 primary 建 stim_unit 连接、拿 unit_id 缓存。
    primary_units = connect_stim_units_to_stim_electrodes(primary_pool, array)
    el2unit = {electrode: unit for electrode, unit in zip(primary_pool, primary_units)}
    primary_diag = build_mapping_diag_from_el2unit(el2unit)
    _raise_if_direct_mapping_has_conflicts(primary_diag)

    cfg["stim_mapping_diagnostics"] = primary_diag

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(3)
    mx.clear_events()

    print(
        f"[SETUP][DUAL] primary pool ready: {len(primary_units)} stim_units active; "
        f"secondary pool ({len(secondary_pool)} electrodes) routed to amplifiers, "
        f"stim_units pending switch."
    )
    return array, primary_units, el2unit


def switch_stim_pool(
    cfg: dict,
    array: mx.Array,
    new_stim_electrodes: list[int],
    label: str,
) -> tuple[list[int], dict[int, int]]:
    """对照组运行时切换 stim 池：完整 select(record + new_stim) + route + connect+query + download。

    用户 2026-05-06 实测肯定确认：``array.download(wells)`` 之前必须连同 cfg 的 record
    电极一起 ``select_electrodes`` + ``route``，不能跳过 record 只声明 stim。
    单独 route 刺激电极后再 download 不会让 record 路由生效。

    切换流程（每一步都对应文档语义）：
      1. ``poweroff_all_stim_units()``                    关旧 stim 输出
      2. ``array.clear_selected_electrodes()``            清旧 selected，避免累积
      3. ``array.select_electrodes(record_electrodes)``   重新声明 record（**关键**）
      4. ``array.select_stimulation_electrodes(new_stim)`` 声明新 stim
      5. ``array.route()``                                record + new_stim 一起 route
                                                          （会清空旧 stim manual switch）
      6. ``connect_electrode_to_stimulation(new_el)`` × N + query → unit_ids
      7. ``array.download(wells)``                        下发硬件
      8. ``configure_and_powerup_stim_units(new_units)``  上电

    切换**不**调（与启动期 setup_routing_and_units 区别）：
      - ``mx.offset()``                              用户指示切换不再做零点校准
      - ``time.sleep(mx.Timing.waitAfterDownload)``  用 protocol 默认 rest 兜稳定
      - ``mx.clear_events()``                        保留 .h5 frame metadata 中已写入的 Event
      - ``array.reset()``                            没必要，clear_selected + 新 select 即可

    返回新池的 stim_units 列表与 electrode->unit 字典。
    """
    config_file = cfg.get("config")
    if not config_file:
        raise RuntimeError(
            "switch_stim_pool requires cfg['config'] to point at a Maxwell .cfg "
            "file so record electrodes can be re-selected before download."
        )

    record_electrodes = extract_cfg_electrodes(config_file)
    if not record_electrodes:
        raise RuntimeError(
            f"Failed to parse any record electrodes from cfg='{config_file}'."
        )

    print(
        f"[SWITCH] '{label}': re-route record={len(record_electrodes)} + "
        f"new_stim={len(new_stim_electrodes)} (no offset/wait/clear_events/reset)"
    )

    # 关掉所有旧 stim_unit 的输出。array 软件层 selected 在 route() 后会被
    # 重置（"empty start"），无需手动 disconnect 旧 stim 桥接。
    poweroff_all_stim_units()

    array.clear_selected_electrodes()
    array.select_electrodes(record_electrodes)
    array.select_stimulation_electrodes(list(new_stim_electrodes))
    array.route()

    # connect_stim_units_to_stim_electrodes 内部先 query_amplifier_at_electrode
    # 校验 routed，再 connect_electrode_to_stimulation + query_stimulation_at_electrode
    # 拿 unit_id；amplifier 路由失败会立即报错。
    new_stim_units = connect_stim_units_to_stim_electrodes(new_stim_electrodes, array)

    # download 把刚 route 的整套配置（含 record + new_stim）下发硬件。
    array.download(cfg["wells"])
    # 不调 mx.Timing.waitAfterDownload sleep / mx.offset / mx.clear_events。

    configure_and_powerup_stim_units(new_stim_units)

    el2unit = {
        electrode: unit
        for electrode, unit in zip(new_stim_electrodes, new_stim_units)
    }

    new_diag = build_mapping_diag_from_el2unit(el2unit)
    if new_diag.get("conflicts"):
        _raise_if_direct_mapping_has_conflicts(new_diag)

    print(
        f"[SWITCH] '{label}' active: {len(new_stim_units)} stim_units; "
        f"electrode->unit pairs: {list(el2unit.items())}"
    )
    return new_stim_units, el2unit


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
