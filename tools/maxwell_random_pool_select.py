"""从 cfg 解析的电极池里随机抽选无冲突的刺激电极组。

实验人员用法：修改文件顶部「修改区」常量后直接运行（不接收命令行参数）：

  /home/maxwell/metaboc-env/bin/python tools/maxwell_random_pool_select.py

两种模式（由顶部常量 MODE 决定）：
  MODE = "experimental"  → 实验组首选：从 cfg 全部 record 电极中抽 32 不冲突
  MODE = "control"       → 对照组：从 cfg record 电极减去 EXCLUDED_ELECTRODES（默认
                            实验组 32 池）后抽 32 不冲突

降级策略：从 TARGET_SIZE 开始试，每个 size 跑最多 MAX_ATTEMPTS 次随机抽样；找到无
冲突的就返回；找不到就降到 size-1，直到 MIN_TARGET_SIZE。

每次 attempt 走完整 select(record) + select_stim + route + **download** + connect +
query 路径——download 必须做：首次 routing 软件层声明的路由只有 download 后才在硬件
层生效，amplifier 路由 / connect_electrode_to_stimulation / query_stimulation_at_electrode
都依赖硬件层配置，否则 query 拿不到稳定 unit_id。

不发刺激、不录音、不 offset；脚本可反复跑用于 probe。

输出：直接 print 一个可粘贴回 experiment_config.py 的 Python 列表 + electrode -> unit
对照对。
"""

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


# =============================================================
# === 实验人员修改区（开始） ===
# =============================================================

# 模式：
#   "experimental" → 实验组首次选 32 不冲突，从 cfg 全部 record 电极抽
#   "control"      → 对照组选 32 不冲突，从 cfg record 电极减 EXCLUDED_ELECTRODES 后抽
MODE = "experimental"

# MODE = "control" 时使用的排除电极组。
#   None：自动从 experiment_config.experimental_stim_electrodes 取（推荐，且与配置文件
#         实验组始终同步）
#   list[int]：手动指定排除列表
EXCLUDED_ELECTRODES: list[int] | None = None

# 目标 size：从 TARGET_SIZE 开始尝试，失败后 size - 1，直到 MIN_TARGET_SIZE
TARGET_SIZE = 32
MIN_TARGET_SIZE = 25

# 每个 size 的最大 random.sample + try 次数
MAX_ATTEMPTS = 20

# 输出几组独立的无冲突池（每组各自抽 + 验证；组间不去重；用于一次跑出多个备选）
N_POOLS = 1

# 随机种子：None 不固定（每次跑结果不同），int 可复现
SEED: int | None = None

# =============================================================
# === 实验人员修改区（结束） ===
# =============================================================


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
    wells: list[int],
) -> tuple[bool, list[int], str]:
    """单次尝试：reset + clear + select_record + select_stim + route + download +
    connect + query。

    download 必须做：首次 routing 软件层声明的路由只有 download 之后才在硬件层
    生效；query_amplifier_at_electrode 与 connect_electrode_to_stimulation 都依赖
    硬件层的 amplifier 路由配置。

    probe 阶段不需要 mx.offset() 与 waitAfterDownload sleep（不发刺激，不录音，
    硬件零点稳定时间无关）。
    """
    array = mx.Array("randompool")
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(record_electrodes)
    array.select_stimulation_electrodes(sampled)
    route_result = array.route()
    if _is_error(route_result):
        return False, [], "array.route() returned error"

    array.download(wells)

    units: list[int] = []
    failed: list[int] = []
    for electrode in sampled:
        unit = _query_unit(array, electrode)
        if unit is None:
            failed.append(electrode)
            units.append(-1)
        else:
            units.append(unit)

    if failed:
        sample = failed[:5]
        suffix = "..." if len(failed) > 5 else ""
        return False, units, f"{len(failed)} electrodes failed amplifier route: {sample}{suffix}"

    unique_units = set(units)
    if len(unique_units) != len(units):
        unit_counts: dict[int, int] = {}
        for u in units:
            unit_counts[u] = unit_counts.get(u, 0) + 1
        conflicts = {u: c for u, c in unit_counts.items() if c > 1}
        return False, units, f"{len(conflicts)} stim_unit conflicts: {conflicts}"

    return True, units, "ok"


def _search_one_pool(
    record_electrodes: list[int],
    candidates: list[int],
    target_size: int,
    min_size: int,
    max_attempts: int,
    pool_index: int,
    wells: list[int],
) -> tuple[int, list[int], list[int]] | None:
    for size in range(target_size, min_size - 1, -1):
        if size > len(candidates):
            print(f"[POOL{pool_index}] size={size} skip (candidates only {len(candidates)})")
            continue
        print(f"\n[POOL{pool_index}] === trying size={size} ===")
        for attempt in range(1, max_attempts + 1):
            sampled = random.sample(candidates, size)
            ok, units, info = _try_pool(record_electrodes, sampled, wells)
            if ok:
                print(f"[POOL{pool_index}] size={size} attempt={attempt} SUCCESS")
                return size, sampled, units
            print(f"[POOL{pool_index}] size={size} attempt={attempt} FAIL: {info}")
    return None


def main() -> int:
    if MODE not in ("experimental", "control"):
        raise ValueError(
            f"MODE must be 'experimental' or 'control'; got {MODE!r}. "
            "Edit the constant at the top of this file."
        )
    if SEED is not None:
        random.seed(SEED)
        print(f"[POOL] random seed = {SEED} (reproducible run)")

    cfg_path = CONFIG.get("config")
    if not cfg_path:
        raise RuntimeError(
            "CONFIG['config'] is empty in experiment_config.py; cannot resolve cfg path."
        )

    record_electrodes = extract_electrodes(cfg_path)
    print(f"[POOL] cfg={cfg_path}")
    print(f"[POOL] record electrodes from cfg: {len(record_electrodes)}")
    print(f"[POOL] MODE={MODE}")

    excluded: set[int] = set()
    if MODE == "control":
        if EXCLUDED_ELECTRODES is None:
            excluded = set(experimental_stim_electrodes)
            print(
                f"[POOL] auto-exclude experimental_stim_electrodes "
                f"({len(excluded)} electrodes from experiment_config.py)"
            )
        else:
            excluded = set(EXCLUDED_ELECTRODES)
            print(f"[POOL] exclude custom EXCLUDED_ELECTRODES ({len(excluded)} electrodes)")

    candidates = [el for el in record_electrodes if el not in excluded]
    print(f"[POOL] candidate pool size: {len(candidates)}")

    if len(candidates) < MIN_TARGET_SIZE:
        raise RuntimeError(
            f"Candidate pool ({len(candidates)}) < MIN_TARGET_SIZE ({MIN_TARGET_SIZE}). "
            "cfg too small or exclude list too large."
        )

    initialize_system()
    mx.activate(CONFIG["wells"])

    successes: list[tuple[int, list[int], list[int]]] = []
    for pool_index in range(1, N_POOLS + 1):
        result = _search_one_pool(
            record_electrodes=record_electrodes,
            candidates=candidates,
            target_size=TARGET_SIZE,
            min_size=MIN_TARGET_SIZE,
            max_attempts=MAX_ATTEMPTS,
            pool_index=pool_index,
            wells=CONFIG["wells"],
        )
        if result is None:
            print(
                f"\n[POOL] pool {pool_index} FAILED in size range "
                f"[{MIN_TARGET_SIZE}, {TARGET_SIZE}] after {MAX_ATTEMPTS} attempts per size."
            )
            continue
        successes.append(result)

    poweroff_all_stim_units()

    if not successes:
        print(
            "\n[POOL] NO POOL FOUND. Try raising MAX_ATTEMPTS, lowering MIN_TARGET_SIZE, "
            "or run mapping_preflight.py with neighbor_retry strategy."
        )
        return 1

    target_var = (
        "experimental_stim_electrodes" if MODE == "experimental" else "control_stim_electrodes"
    )

    print("\n" + "=" * 70)
    for index, (size, electrodes, units) in enumerate(successes, start=1):
        print(f"[POOL {index}/{len(successes)}] {size} non-conflicting electrodes")
        print(f"[POOL {index}] paste to experiment_config.py {target_var}:")
        print(electrodes)
        print(f"[POOL {index}] electrode -> stim_unit pairs:")
        print(list(zip(electrodes, units)))
        print("-" * 70)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
