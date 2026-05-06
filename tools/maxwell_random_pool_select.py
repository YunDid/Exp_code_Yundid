"""从 cfg 解析的电极池里随机抽选无冲突的刺激电极组。

正式实验前用：选实验组 32 池子时跑模式 A，选对照组 32 池子（与实验组完全不重叠）时
跑模式 B。脚本不发刺激、不录音、不 download；只走 select+route+connect+query 验证
随机抽样的 32 电极是否互不冲突。

两种用法：

  # 模式 A — 任意首选：从 cfg 全部 record 电极中选 32 个不冲突的
  /home/maxwell/metaboc-env/bin/python tools/maxwell_random_pool_select.py

  # 模式 B — 对照组：选与实验组 experimental_stim_electrodes 完全不重叠的另一组 32 个
  /home/maxwell/metaboc-env/bin/python tools/maxwell_random_pool_select.py \
      --exclude experimental

  # 模式 B' — 对照组（显式排除列表）
  /home/maxwell/metaboc-env/bin/python tools/maxwell_random_pool_select.py \
      --exclude "304,1019,2024,..."

降级策略：从 --target（默认 32）试起，每个 size 跑最多 --attempts（默认 20）次随机
抽样，找到无冲突的就返回；找不到就降到 size-1，直到 --min（默认 25）。

输出：直接 print 一个可粘贴回 experiment_config.py 的 Python 列表 + 每个电极对应的
stim_unit；可选 --n-pools N 输出 N 组（每组都无冲突，组间不去重）。

要求：mxwserver 已启动；cfg 路径在 CONFIG["config"] 或 --cfg 指定。
"""

import argparse
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXP_CODE_DIR = THIS_DIR.parent
if str(EXP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_CODE_DIR))

import maxlab as mx

from cfg_utils import extract_electrodes
from experiment_config import CONFIG, experimental_stim_electrodes
from system_api import initialize_system, poweroff_all_stim_units


DEFAULT_TARGET_SIZE = 32
DEFAULT_MIN_TARGET_SIZE = 25
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_N_POOLS = 1


def _is_error(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() == "error"


def _has_routed_amplifier(array: mx.Array, electrode: int) -> bool:
    amp = array.query_amplifier_at_electrode(electrode)
    if _is_error(amp):
        return False
    return len(str(amp).strip()) > 0


def _query_unit(array: mx.Array, electrode: int) -> int | None:
    """connect + query 拿 stim_unit；失败返回 None。"""
    if not _has_routed_amplifier(array, electrode):
        return None
    connect_result = array.connect_electrode_to_stimulation(electrode)
    if _is_error(connect_result):
        return None
    stim = array.query_stimulation_at_electrode(electrode)
    if _is_error(stim):
        return None
    text = str(stim).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _try_pool(
    record_electrodes: list[int],
    sampled: list[int],
) -> tuple[bool, list[int], str]:
    """单次尝试：select+route+connect+query；返回 (success, units, info)。

    每次都在新 mx.Array 上做 reset+clear 重新建路由，单次不污染下次状态。
    """
    array = mx.Array("randompool")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(record_electrodes)
    array.select_stimulation_electrodes(sampled)
    route_result = array.route()
    if _is_error(route_result):
        return False, [], "array.route() returned error"

    units: list[int] = []
    failed: list[int] = []
    for el in sampled:
        unit = _query_unit(array, el)
        if unit is None:
            failed.append(el)
            units.append(-1)
        else:
            units.append(unit)

    if failed:
        return False, units, f"{len(failed)} electrodes failed amplifier route: {failed[:5]}{'...' if len(failed) > 5 else ''}"

    unique_units = set(units)
    if len(unique_units) != len(units):
        unit_counts: dict[int, int] = {}
        for u in units:
            unit_counts[u] = unit_counts.get(u, 0) + 1
        conflicts = {u: c for u, c in unit_counts.items() if c > 1}
        return False, units, f"{len(conflicts)} stim_unit conflicts: {conflicts}"

    return True, units, "ok"


def search_one_pool(
    record_electrodes: list[int],
    candidates: list[int],
    target_size: int,
    min_size: int,
    max_attempts: int,
    pool_index: int = 0,
) -> tuple[int, list[int], list[int]] | None:
    """从 candidates 抽，在 [min_size, target_size] 区间内找一组无冲突；返回 (size, electrodes, units) 或 None。"""
    for size in range(target_size, min_size - 1, -1):
        if size > len(candidates):
            print(f"[POOL{pool_index}] size={size} skip (candidates pool only {len(candidates)})")
            continue
        print(f"\n[POOL{pool_index}] === trying size={size} ===")
        for attempt in range(1, max_attempts + 1):
            sampled = random.sample(candidates, size)
            ok, units, info = _try_pool(record_electrodes, sampled)
            if ok:
                print(f"[POOL{pool_index}] size={size} attempt={attempt} SUCCESS")
                return size, sampled, units
            print(f"[POOL{pool_index}] size={size} attempt={attempt} FAIL: {info}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Random select non-conflicting stim electrode pool from cfg."
    )
    parser.add_argument(
        "--cfg",
        default=None,
        help="cfg path; default uses CONFIG['config'] from experiment_config.py.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help='Exclude from candidates: "experimental" (auto from experimental_stim_electrodes) '
        'or comma-separated electrode IDs like "304,1019,2024".',
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Initial target size; will degrade down to --min. Default {DEFAULT_TARGET_SIZE}.",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_TARGET_SIZE,
        help=f"Minimum acceptable size; default {DEFAULT_MIN_TARGET_SIZE}.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Max random attempts per size; default {DEFAULT_MAX_ATTEMPTS}.",
    )
    parser.add_argument(
        "--n-pools",
        type=int,
        default=DEFAULT_N_POOLS,
        help=f"Output how many independent non-conflicting pools; default {DEFAULT_N_POOLS}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg_path = args.cfg or CONFIG.get("config")
    if not cfg_path:
        raise RuntimeError(
            "No cfg path provided; set CONFIG['config'] in experiment_config.py or pass --cfg."
        )

    record_electrodes = extract_electrodes(cfg_path)
    print(f"[POOL] cfg={cfg_path}")
    print(f"[POOL] record electrodes from cfg: {len(record_electrodes)}")

    excluded: set[int] = set()
    exclude_label = "none"
    if args.exclude:
        if args.exclude == "experimental":
            excluded = set(experimental_stim_electrodes)
            exclude_label = f"experimental_stim_electrodes ({len(excluded)} electrodes)"
        else:
            excluded = {int(s.strip()) for s in args.exclude.split(",") if s.strip()}
            exclude_label = f"custom list ({len(excluded)} electrodes)"
    print(f"[POOL] exclude: {exclude_label}")

    candidates = [el for el in record_electrodes if el not in excluded]
    print(f"[POOL] candidate pool after exclude: {len(candidates)} electrodes")

    if len(candidates) < args.min:
        raise RuntimeError(
            f"Candidate pool ({len(candidates)}) is smaller than --min ({args.min}). "
            "cfg too small or exclude list too large."
        )

    initialize_system()
    mx.activate(CONFIG["wells"])

    successes: list[tuple[int, list[int], list[int]]] = []
    for pool_index in range(1, args.n_pools + 1):
        result = search_one_pool(
            record_electrodes=record_electrodes,
            candidates=candidates,
            target_size=args.target,
            min_size=args.min,
            max_attempts=args.attempts,
            pool_index=pool_index,
        )
        if result is None:
            print(
                f"\n[POOL] pool {pool_index} FAILED to find any non-conflicting set "
                f"in [{args.min}, {args.target}] within {args.attempts} attempts each."
            )
            continue
        successes.append(result)

    poweroff_all_stim_units()

    if not successes:
        print(
            "\n[POOL] NO POOL FOUND. Try increasing --attempts, lowering --min, "
            "or run mapping_preflight.py with neighbor_retry strategy on a manually picked seed pool."
        )
        return 1

    print("\n" + "=" * 70)
    for index, (size, electrodes, units) in enumerate(successes, start=1):
        print(f"[POOL {index}/{len(successes)}] {size} non-conflicting electrodes")
        print(f"[POOL {index}] electrodes (paste to experiment_config.py):")
        print(electrodes)
        print(f"[POOL {index}] electrode -> stim_unit pairs:")
        print(list(zip(electrodes, units)))
        print("-" * 70)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
