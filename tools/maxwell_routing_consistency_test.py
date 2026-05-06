"""Maxwell routing 输入集合一致性实测工具（冲突状态对照版）。

要回答的核心问题：
  「一组 stim 电极在全候选池 routing 下被算法挑成『无冲突』，把这组电极单独
  select 做窄 routing，是否仍然无冲突？」

unit ID 在不同 routing 下变没变不重要——每个电极重新 query 即可。
关键是是否**有两个电极撞到同一个 unit**，即冲突状态是否在 routing 输入集合
变化下保持。

四个 scenario：
  A1_narrow_N        : select N target 第一次跑
  B_expanded_r50     : select target + 邻居 r 半径（几百），仍 query 同一组 N target
  C_subset_K         : select 前 K 个 target
  A2_narrow_N_repeat : select N target 第二次跑（与 A1 比较复现性）

每个 scenario 立即报告冲突状态：unique_units 数 vs queried 数；列出冲突组。

三个对照维度：
  复现性     A1 vs A2 ：unit ID + 冲突状态是否完全相同
  扩容敏感性 A1 vs B  ：unit ID + 冲突状态是否随 select 集合扩容变化
  缩小稳定性 A1 vs C  ：C 子集里那 K 个 target 的 unit ID + 冲突状态是否与 A1 相同

输入 target 来源（任选一种，按优先级）：
  --preflight-json PATH ：从 mapping_preflight 输出 JSON 读
                          meta.mapping_preflight.resolved_electrodes
                          （强烈推荐：直接验证 preflight 产出在窄 routing 下是否仍成立）
  --targets-csv "a,b,c" ：命令行传逗号分隔列表
  （都不传）            ：使用 DEFAULT_TARGET_STIM_ELECTRODES（与 mapping_preflight.py 同步）

跑法（Linux 端，必须用 venv Python）：
  # 验证 preflight 产出
  /home/maxwell/metaboc-env/bin/python tools/maxwell_routing_consistency_test.py \\
      --cfg /home/maxwell/configs/your_recording.cfg \\
      --preflight-json /path/to/mapping_preflight_20260506_HHMMSS.json

  # 验证原始 target（默认 DEFAULT_TARGET_STIM_ELECTRODES）
  /home/maxwell/metaboc-env/bin/python tools/maxwell_routing_consistency_test.py \\
      --cfg /home/maxwell/configs/your_recording.cfg

权威性：query_stimulation_at_electrode 直接 raw 字符串对照，按 [[Maxwell -
query_stimulation_at_electrode 返回字符串解析陷阱]] 修复方式 str(stim).strip()
不切片。本脚本不调 array.download，不发刺激，不录制——可以反复跑。
"""

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXP_CODE_DIR = THIS_DIR.parent
if str(EXP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_CODE_DIR))

import maxlab as mx

from cfg_utils import extract_electrodes
from system_api import expand_stim_electrode_pool, initialize_system


# 默认目标 stim 电极列表（与 mapping_preflight.py main 同步）
DEFAULT_TARGET_STIM_ELECTRODES = [
    304, 1019, 2024, 2749, 3838, 7346, 5984, 6076,
    7963, 10482, 8461, 10506, 18617, 12312, 13166, 14097,
    12630, 16489, 17444, 13755, 13369, 18462, 18653, 19678,
    20524, 20717, 20506, 22772, 23035, 23204, 24739, 25563,
]
DEFAULT_NEIGHBOR_RADIUS = 50
DEFAULT_SUBSET_SIZE = 10
DEFAULT_WELLS = [0]


def _is_error_or_empty(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "error"


def _query_unit_raw(array: mx.Array, electrode: int) -> str | None:
    """Return the raw string returned by query_stimulation_at_electrode, or None."""
    stim = array.query_stimulation_at_electrode(electrode)
    if _is_error_or_empty(stim):
        return None
    return str(stim).strip()


def _conflict_summary(unit_by_target: "dict[int, str | None]") -> dict:
    """Compute conflict status for a scenario's query result.

    Returns:
        queried       : 总查询数（包括 query 失败 None）
        success       : 成功拿到 unit 的电极数
        unique_units  : 不同 unit 的数量
        conflicts     : {unit: [el_a, el_b, ...]}（unit 被多个电极共占的情况）
    """
    units_seen: list[str] = []
    unit_to_els: dict[str, list[int]] = {}
    for el, u in unit_by_target.items():
        if u is None:
            continue
        units_seen.append(u)
        unit_to_els.setdefault(u, []).append(el)

    conflicts = {u: sorted(els) for u, els in unit_to_els.items() if len(els) > 1}

    return {
        "queried": len(unit_by_target),
        "success": len(units_seen),
        "unique_units": len(set(units_seen)),
        "conflicts": conflicts,
    }


def _print_scenario_conflict(name: str, summary: dict) -> None:
    n_conflict_groups = len(summary["conflicts"])
    extra = sum(len(els) - 1 for els in summary["conflicts"].values())
    if n_conflict_groups == 0 and summary["success"] == summary["queried"]:
        print(
            f"[CONFLICT] {name}: {summary['queried']} target → "
            f"{summary['unique_units']} unique units → 无冲突 ✓"
        )
    elif n_conflict_groups == 0:
        print(
            f"[CONFLICT] {name}: {summary['queried']} target ({summary['success']} 查到) → "
            f"{summary['unique_units']} unique units → 无冲突但有 {summary['queried'] - summary['success']} 个 query 失败"
        )
    else:
        print(
            f"[CONFLICT] {name}: {summary['queried']} target ({summary['success']} 查到) → "
            f"{summary['unique_units']} unique units → ✗ {n_conflict_groups} 个冲突组 "
            f"({extra} 个 extra electrodes)"
        )
        for unit, els in summary["conflicts"].items():
            print(f"[CONFLICT]   unit={unit} 被 {els} 共占")


def _run_scenario(
    name: str,
    record_electrodes: list[int],
    stim_select_set: list[int],
    query_targets: list[int],
) -> "dict[int, str | None]":
    """Build a fresh routing scenario and query unit for each target in select_set."""
    print(f"\n[CONSISTENCY] === scenario: {name} (stim_select_count={len(stim_select_set)}) ===")

    array = mx.Array(f"consistency_{name}")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(record_electrodes)
    array.select_stimulation_electrodes(stim_select_set)
    array.route()

    select_set = set(stim_select_set)
    queryable = [el for el in query_targets if el in select_set]

    unit_by_target: dict[int, str | None] = {}
    connected: list[int] = []

    for el in queryable:
        amp = array.query_amplifier_at_electrode(el)
        if _is_error_or_empty(amp):
            print(f"[CONSISTENCY]   skip electrode {el}: amplifier not routed")
            unit_by_target[el] = None
            continue
        connect_result = array.connect_electrode_to_stimulation(el)
        if _is_error_or_empty(connect_result):
            print(f"[CONSISTENCY]   skip electrode {el}: connect_electrode_to_stimulation failed")
            unit_by_target[el] = None
            continue
        connected.append(el)
        unit_by_target[el] = _query_unit_raw(array, el)

    for el in queryable:
        unit = unit_by_target.get(el)
        print(f"[CONSISTENCY]   electrode {el} -> unit {unit}")

    summary = _conflict_summary(unit_by_target)
    _print_scenario_conflict(name, summary)

    # cleanup
    for el in connected:
        array.disconnect_electrode_from_stimulation(el)
    try:
        array.close()
    except Exception:
        pass

    return unit_by_target


def _print_full_table(
    targets: list[int],
    results: "dict[str, dict[int, str | None]]",
) -> None:
    scenario_names = list(results.keys())
    col_width = max(len(name) for name in scenario_names) + 2
    header = f"{'electrode':>10} | " + " | ".join(f"{name:>{col_width}}" for name in scenario_names)
    print("\n[CONSISTENCY] ========== 全量对照表 ==========")
    print(header)
    print("-" * len(header))

    for el in targets:
        row_units: list[str] = []
        for name in scenario_names:
            if el in results[name]:
                row_units.append(str(results[name][el]))
            else:
                row_units.append("—")
        seen = [u for u in row_units if u != "—"]
        is_diff = len(set(seen)) > 1
        row_str = f"{el:>10} | " + " | ".join(f"{u:>{col_width}}" for u in row_units)
        if is_diff:
            row_str += "  <- DIFF"
        print(row_str)


def _print_conflict_table(
    results: "dict[str, dict[int, str | None]]",
) -> "dict[str, dict]":
    """打印每个 scenario 的冲突状态对照，返回 {scenario_name: summary}。"""
    print("\n[CONSISTENCY] ========== 冲突状态对照（核心维度）==========")
    summaries: dict[str, dict] = {}
    for name, unit_by_target in results.items():
        s = _conflict_summary(unit_by_target)
        summaries[name] = s
        n_groups = len(s["conflicts"])
        status = "✓ 无冲突" if n_groups == 0 else f"✗ {n_groups} 冲突组"
        print(
            f"[CONSISTENCY] {name:>30}: queried={s['queried']:>3}  "
            f"unique={s['unique_units']:>3}  status={status}"
        )
    return summaries


def _compare_pair(
    results: "dict[str, dict[int, str | None]]",
    summaries: "dict[str, dict]",
    scenario_a: str,
    scenario_b: str,
    label: str,
) -> dict:
    """Compare two scenarios on overlapping (queried in both) targets."""
    common = set(results[scenario_a].keys()) & set(results[scenario_b].keys())
    unit_diffs: list[tuple[int, str | None, str | None]] = []
    for el in sorted(common):
        a = results[scenario_a][el]
        b = results[scenario_b][el]
        if a != b:
            unit_diffs.append((el, a, b))

    conflict_a = len(summaries[scenario_a]["conflicts"]) > 0
    conflict_b = len(summaries[scenario_b]["conflicts"]) > 0
    conflict_status_changed = conflict_a != conflict_b

    print(f"\n[CONSISTENCY] -- {label} ({scenario_a} vs {scenario_b}) --")
    print(f"[CONSISTENCY] 共同 target 数 = {len(common)}")
    print(
        f"[CONSISTENCY] 冲突状态：{scenario_a}={'冲突' if conflict_a else '无冲突'}  "
        f"{scenario_b}={'冲突' if conflict_b else '无冲突'}  "
        f"{'⚠ 变化' if conflict_status_changed else '✓ 一致'}"
    )
    if unit_diffs:
        print(f"[CONSISTENCY] unit ID 不一致 = {len(unit_diffs)} 个：")
        for el, a, b in unit_diffs:
            print(f"[CONSISTENCY]   electrode {el}: {scenario_a}={a}, {scenario_b}={b}")
    else:
        print(f"[CONSISTENCY] unit ID 全部一致")

    return {
        "common": len(common),
        "unit_diffs": unit_diffs,
        "conflict_status_changed": conflict_status_changed,
    }


def _print_overall_conclusion(
    repro_cmp: dict,
    expand_cmp: dict,
    shrink_cmp: dict,
) -> None:
    print("\n[CONSISTENCY] ========== 总结 ==========")

    repro_unit_ok = not repro_cmp["unit_diffs"]
    repro_conflict_ok = not repro_cmp["conflict_status_changed"]
    expand_conflict_ok = not expand_cmp["conflict_status_changed"]
    shrink_conflict_ok = not shrink_cmp["conflict_status_changed"]

    print(f"[CONSISTENCY] 复现性     unit ID 一致 = {'✓' if repro_unit_ok else '✗'}    冲突状态一致 = {'✓' if repro_conflict_ok else '⚠'}")
    print(f"[CONSISTENCY] 扩容敏感性  冲突状态保持 = {'✓ 全候选池无冲突 → 窄集合也无冲突' if expand_conflict_ok else '⚠ 冲突状态在两 routing 下不同'}")
    print(f"[CONSISTENCY] 缩小稳定性  冲突状态保持 = {'✓' if shrink_conflict_ok else '⚠'}")

    print()

    if not (repro_unit_ok and repro_conflict_ok):
        print("[CONSISTENCY] ⚠ 复现性失败：同输入两次跑得到不同结果。routing 算法可能含随机性。")
        print("[CONSISTENCY]   先解决这个，其他对照才有意义。")
        return

    if expand_conflict_ok and shrink_conflict_ok:
        print(
            "[CONSISTENCY] ✓ 核心问题答案：在本 cfg + 本 target 集合下，"
            "在全候选池 routing 下不冲突的电极组，单独 select 做窄 routing 时**仍然不冲突**。"
        )
        print(
            "[CONSISTENCY]   推论：mapping_preflight 在扩容 routing 下挑出的 resolved_electrodes "
            "可直接用于正式实验，final_route_check 是冗余复测（保留作 belt-and-suspenders）。"
        )
    else:
        print(
            "[CONSISTENCY] ⚠ 核心问题答案：在全候选池 routing 下不冲突的电极组，"
            "改成窄 select 时**冲突状态会变**。"
        )
        if not expand_conflict_ok:
            print("[CONSISTENCY]   - A1（窄）vs B（扩容）：冲突状态不同")
        if not shrink_conflict_ok:
            print("[CONSISTENCY]   - A1（窄）vs C（子集）：冲突状态不同")
        print(
            "[CONSISTENCY]   推论：preflight 算法的产出在窄 routing 下不再保证无冲突；"
            "mapping_preflight 必须用 final_route_check 在 resolved 窄集合下复测，"
            "并以复测结果决定是否进入正式实验。这是 mapping_preflight.py 当前的实现。"
        )

    if expand_cmp["unit_diffs"] or shrink_cmp["unit_diffs"]:
        n = len(expand_cmp["unit_diffs"]) + len(shrink_cmp["unit_diffs"])
        print(
            f"[CONSISTENCY] 注：unit ID 在不同 routing 下有 {n} 个差异点，"
            "但这不影响实验（每个电极在正式 routing 下重新 query unit 即可）。"
        )


def _load_targets(args: argparse.Namespace) -> list[int]:
    """按优先级加载 target 列表：preflight-json > targets-csv > DEFAULT。"""
    if args.preflight_json:
        with open(args.preflight_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            resolved = data["meta"]["mapping_preflight"]["resolved_electrodes"]
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"无法从 {args.preflight_json} 提取 meta.mapping_preflight.resolved_electrodes：{exc}"
            )
        targets = [int(e) for e in resolved]
        print(f"[CONSISTENCY] target 来源：mapping_preflight 输出 JSON ({args.preflight_json})")
        return targets

    if args.targets_csv:
        targets = [int(s.strip()) for s in args.targets_csv.split(",") if s.strip()]
        print(f"[CONSISTENCY] target 来源：命令行 --targets-csv")
        return targets

    print("[CONSISTENCY] target 来源：DEFAULT_TARGET_STIM_ELECTRODES（脚本顶部硬编码）")
    return list(DEFAULT_TARGET_STIM_ELECTRODES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maxwell routing 输入集合一致性测试（4 scenarios + 冲突状态对照）",
    )
    parser.add_argument("--cfg", required=True, help="MaxLab cfg 文件路径（提供 record electrodes）")
    parser.add_argument(
        "--preflight-json",
        help="读 mapping_preflight 输出 JSON 中 meta.mapping_preflight.resolved_electrodes 作为 target",
    )
    parser.add_argument(
        "--targets-csv",
        help='命令行传 target 列表，例：--targets-csv "304,1019,2024,..."',
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=DEFAULT_NEIGHBOR_RADIUS,
        help=f"扩容 scenario B 的邻居半径（默认 {DEFAULT_NEIGHBOR_RADIUS}）",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help=f"缩小 scenario C 的 target 子集大小（默认 {DEFAULT_SUBSET_SIZE}，从 target 头部取）",
    )
    args = parser.parse_args()

    targets = _load_targets(args)
    cfg_path = args.cfg

    print(f"[CONSISTENCY] cfg={cfg_path}")
    print(f"[CONSISTENCY] target count={len(targets)}")
    print(f"[CONSISTENCY] neighbor radius={args.neighbor_radius}")
    print(f"[CONSISTENCY] subset size={args.subset_size}")

    if args.subset_size > len(targets):
        print(
            f"[CONSISTENCY] ⚠ subset_size ({args.subset_size}) > target count ({len(targets)})，"
            f"已自动调整为 {len(targets)}"
        )
        args.subset_size = len(targets)

    record_electrodes = extract_electrodes(cfg_path)
    print(f"[CONSISTENCY] record electrodes count={len(record_electrodes)}")

    initialize_system()
    mx.activate(DEFAULT_WELLS)

    results: "dict[str, dict[int, str | None]]" = {}

    n = len(targets)
    name_a1 = f"A1_narrow_{n}"
    name_a2 = f"A2_narrow_{n}_repeat"
    name_b = f"B_expanded_r{args.neighbor_radius}"
    name_c = f"C_subset_{args.subset_size}"

    # A1：N target 第一次跑
    results[name_a1] = _run_scenario(name_a1, record_electrodes, targets, targets)

    # B：扩容池
    expanded = expand_stim_electrode_pool(targets, args.neighbor_radius)
    results[name_b] = _run_scenario(name_b, record_electrodes, expanded, targets)

    # C：前 K 子集
    subset_targets = targets[: args.subset_size]
    results[name_c] = _run_scenario(name_c, record_electrodes, subset_targets, targets)

    # A2：N target 第二次跑
    results[name_a2] = _run_scenario(name_a2, record_electrodes, targets, targets)

    _print_full_table(targets, results)
    summaries = _print_conflict_table(results)

    print("\n[CONSISTENCY] ========== 维度对照 ==========")
    repro_cmp = _compare_pair(results, summaries, name_a1, name_a2, "复现性（同输入跑两次）")
    expand_cmp = _compare_pair(results, summaries, name_a1, name_b, "扩容敏感性（窄 vs 扩容）")
    shrink_cmp = _compare_pair(results, summaries, name_a1, name_c, "缩小稳定性（窄 vs 子集）")

    _print_overall_conclusion(repro_cmp, expand_cmp, shrink_cmp)


if __name__ == "__main__":
    main()
