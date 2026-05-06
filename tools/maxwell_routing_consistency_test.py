"""Maxwell routing 输入集合一致性实测工具。

目标：验证 select_stimulation_electrodes 输入集合变化是否会让同一 stim 电极
被分配到不同的 stim_unit。这是 mapping_preflight 算法依赖的核心假设之一——
preflight 用扩容候选池 routing 做 probe，正式实验用窄集合 routing；
如果两个 routing 输入集合下同一电极的 unit 分配不同，preflight 拿到的
unit 映射就不能直接用作正式实验的权威值（必须 final_route_check 复测）。

方法：
  Scenario A (narrow_32) : select(target 32 stim 电极) → route → connect 全部
                           target → 逐个 query stim_unit
  Scenario B (expanded_r): select(target 32 + 邻居 r 半径候选池 = 几百个) →
                           route → connect 全部 target → 逐个 query stim_unit
  对照报告：每个 target 在 A 与 B 下的 unit 是否一致；DIFF 数即结论

跑法（Linux 端，必须用 venv Python）：
  /home/maxwell/metaboc-env/bin/python tools/maxwell_routing_consistency_test.py \\
      --cfg /home/maxwell/configs/your_recording.cfg

权威性：用 query_stimulation_at_electrode 的真实硬件返回值做对照，没有任何缓存
/ 中间计算。读取返回值时按 [[Maxwell - query_stimulation_at_electrode 返回字符串
解析陷阱]] 的修复方式直接 int(stim) / str(stim) 不切片。

注意：
- 本脚本不调 array.download，仅用 routing + connect_electrode_to_stimulation 状态
  下的 query 拿 unit_id（与 stimulate.html line 533-548 官方写法一致）
- 不发刺激、不录制、不修改任何持久化状态——可以反复跑
"""

import argparse
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
DEFAULT_WELLS = [0]


def _is_error_or_empty(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "error"


def _query_unit_raw(array: mx.Array, electrode: int) -> str | None:
    """Return the raw string returned by query_stimulation_at_electrode, or None.

    保留原始字符串形式（不 int 转换），便于看 API 实际返回值；下游对照只
    比较字符串相等性，避开 int 解析的所有陷阱。
    """
    stim = array.query_stimulation_at_electrode(electrode)
    if _is_error_or_empty(stim):
        return None
    return str(stim).strip()


def _build_routing_and_connect_targets(
    record_electrodes: list[int],
    stim_select_set: list[int],
    targets: list[int],
    array_name: str,
) -> tuple[mx.Array, list[int]]:
    """Build a fresh routing scenario and connect every target to stimulation.

    返回 (array, 成功 connect 的 target 列表)。target 在该 scenario 不能 connect 时
    在 unit map 里记 None，对照报告也能反映这种"该 routing 下连 amp 都不 routed"。
    """
    array = mx.Array(array_name)
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(record_electrodes)
    array.select_stimulation_electrodes(stim_select_set)
    array.route()

    connected: list[int] = []
    for el in targets:
        amp = array.query_amplifier_at_electrode(el)
        if _is_error_or_empty(amp):
            print(f"[CONSISTENCY]   skip electrode {el}: amplifier not routed in this scenario")
            continue
        connect_result = array.connect_electrode_to_stimulation(el)
        if _is_error_or_empty(connect_result):
            print(f"[CONSISTENCY]   skip electrode {el}: connect_electrode_to_stimulation failed")
            continue
        connected.append(el)
    return array, connected


def _run_scenario(
    name: str,
    record_electrodes: list[int],
    stim_select_set: list[int],
    targets: list[int],
) -> dict[int, str | None]:
    print(f"\n[CONSISTENCY] === scenario: {name} (stim_select_count={len(stim_select_set)}) ===")
    array, connected = _build_routing_and_connect_targets(
        record_electrodes,
        stim_select_set,
        targets,
        array_name=f"consistency_{name}",
    )
    unit_by_target: dict[int, str | None] = {el: None for el in targets}
    for el in connected:
        unit_by_target[el] = _query_unit_raw(array, el)
    for el in targets:
        unit = unit_by_target[el]
        print(f"[CONSISTENCY]   electrode {el} -> unit {unit}")

    # cleanup：disconnect 全部 connected target，并 close array，准备下一 scenario
    for el in connected:
        array.disconnect_electrode_from_stimulation(el)
    try:
        array.close()
    except Exception:
        pass
    return unit_by_target


def _print_diff_report(
    targets: list[int],
    results: "dict[str, dict[int, str | None]]",
) -> list[int]:
    scenario_names = list(results.keys())
    col_width = max(len(name) for name in scenario_names) + 2
    header = f"{'electrode':>10} | " + " | ".join(f"{name:>{col_width}}" for name in scenario_names)
    print("\n[CONSISTENCY] ========== 对照报告 ==========")
    print(header)
    print("-" * len(header))

    differences: list[int] = []
    for el in targets:
        row_units = [str(results[name].get(el)) for name in scenario_names]
        unique = set(row_units)
        is_diff = len(unique) > 1
        if is_diff:
            differences.append(el)
        row_str = f"{el:>10} | " + " | ".join(f"{u:>{col_width}}" for u in row_units)
        if is_diff:
            row_str += "  <- DIFF"
        print(row_str)
    return differences


def _print_conclusion(differences: list[int], total: int) -> None:
    print("\n[CONSISTENCY] ========== 结论 ==========")
    if differences:
        print(
            f"[CONSISTENCY] {len(differences)} / {total} 个 target 在不同 routing "
            f"输入集合下 stim_unit 不一致："
        )
        print(f"[CONSISTENCY]   {differences}")
        print(
            "[CONSISTENCY] 证明：routing 输入集合变化会改变 stim_unit 分配。"
        )
        print(
            "[CONSISTENCY] 推论：preflight 拿到的扩容 routing unit 与正式实验窄集合 routing "
            "unit 不能等同；mapping_preflight 必须用窄集合复测 (final_route_check) 后取窄集合 "
            "unit 为权威——这正是 mapping_preflight.py 当前的实现。"
        )
    else:
        print(
            f"[CONSISTENCY] 所有 {total} 个 target 在两种 routing 下 unit 完全一致。"
        )
        print(
            "[CONSISTENCY] 证明：在本 cfg + 本 target 集合下，maxlab routing 算法对 "
            "select_stimulation_electrodes 输入集合变化稳定。"
        )
        print(
            "[CONSISTENCY] 推论：preflight 扩容 routing 拿到的 unit 可直接用于正式实验；"
            "final_route_check 在本场景下是冗余复测（保留作为 belt-and-suspenders）。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maxwell routing 输入集合一致性测试",
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="MaxLab cfg 文件路径（提供 record electrodes）",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=DEFAULT_NEIGHBOR_RADIUS,
        help=f"扩容 scenario 的邻居半径（默认 {DEFAULT_NEIGHBOR_RADIUS}）",
    )
    args = parser.parse_args()

    targets = list(DEFAULT_TARGET_STIM_ELECTRODES)
    cfg_path = args.cfg

    print(f"[CONSISTENCY] cfg={cfg_path}")
    print(f"[CONSISTENCY] target count={len(targets)}")
    print(f"[CONSISTENCY] neighbor radius={args.neighbor_radius}")

    record_electrodes = extract_electrodes(cfg_path)
    print(f"[CONSISTENCY] record electrodes count={len(record_electrodes)}")

    initialize_system()
    mx.activate(DEFAULT_WELLS)

    results: "dict[str, dict[int, str | None]]" = {}

    # Scenario A：仅 32 target（窄集合，与正式实验输入一致）
    results["narrow_32"] = _run_scenario(
        "narrow_32",
        record_electrodes,
        targets,
        targets,
    )

    # Scenario B：扩容池（target + 邻居 r 半径，与 mapping_preflight probe routing 一致）
    expanded = expand_stim_electrode_pool(targets, args.neighbor_radius)
    scenario_b_name = f"expanded_r{args.neighbor_radius}"
    results[scenario_b_name] = _run_scenario(
        scenario_b_name,
        record_electrodes,
        expanded,
        targets,
    )

    differences = _print_diff_report(targets, results)
    _print_conclusion(differences, len(targets))


if __name__ == "__main__":
    main()
