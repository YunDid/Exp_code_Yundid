"""从 cfg 解析的电极池里随机抽选无冲突的刺激电极组。

实验人员用法：修改文件顶部「修改区」常量后直接运行（不接收命令行参数）：

  /home/maxwell/metaboc-env/bin/python tools/maxwell_random_pool_select.py

两种模式（由顶部常量 MODE 决定）：
  MODE = "experimental"  → 实验组首选：从 cfg 全部 record 电极中抽 32 不冲突
  MODE = "control"       → 对照组：从 cfg record 电极减去 EXCLUDED_ELECTRODES（默认
                            实验组 32 池）后抽 32 不冲突

抽样策略：**分层抽样**（stratified sampling），candidates 按 electrode ID 排序后切成
target_size 个区间，每个区间随机抽 1 个，保证空间分散。距离大的电极更不容易共用同
一个 stim_unit；如果第一次仍冲突，再做"冲突替换"：把 unit 重复的电极移除，从未抽过
的 candidates 中拉远距离补一个，重新 try。

降级策略：从 TARGET_SIZE 开始试，每个 size 跑最多 MAX_ATTEMPTS 次抽样；找到无冲突
的就返回；找不到就降到 size-1，直到 MIN_TARGET_SIZE。

每次 attempt 走完整 select(record) + select_stim + route + **download** + connect +
query 路径——download 必须做：首次 routing 软件层声明的路由只有 download 后才在硬件
层生效，amplifier 路由 / connect_electrode_to_stimulation / query_stimulation_at_electrode
都依赖硬件层配置，否则 query 拿不到稳定 unit_id。

不发刺激、不录音、不 offset；脚本可反复跑用于 probe。

输出：清晰分两段——
  1. 一行 Python 列表字面量：直接粘贴回 experiment_config.py
  2. electrode -> stim_unit 对照表：每行一对，便于交叉校验
"""

import json
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


def _stratified_sample(candidates: list[int], target_size: int) -> list[int]:
    """分层抽样：candidates 排序后切成 target_size 个区间，每个区间随机抽 1 个。

    理由：electrode_id 在 MaxOne 上对应物理 row/col（id // 220 = row, id % 220 = col），
    ID 排序近似行扫描顺序；ID 空间分层等价于物理空间分散，避免相邻电极扎堆共享 stim_unit。

    退化：target_size > len(candidates) 时降为 random.sample；区间内随机抽到的 idx
    与之前重复时取下一个未取过的。
    """
    if target_size > len(candidates):
        return random.sample(candidates, len(candidates))

    candidates_sorted = sorted(candidates)
    n = len(candidates_sorted)
    bucket = n / target_size

    sampled_indices: set[int] = set()
    sampled: list[int] = []
    for i in range(target_size):
        start = int(i * bucket)
        end = max(start + 1, int((i + 1) * bucket))
        end = min(end, n)
        # 区间内随机选一个未选过的
        idx = random.randint(start, end - 1)
        if idx in sampled_indices:
            # 在区间内顺序找下一个未选；若区间已满则跨到下一区间末
            for cand_idx in range(start, n):
                if cand_idx not in sampled_indices:
                    idx = cand_idx
                    break
        sampled_indices.add(idx)
        sampled.append(candidates_sorted[idx])

    # 极少情况下 sampled_indices 内部冲突没补齐，用全集兜底补足
    if len(sampled) < target_size:
        for cand_idx in range(n):
            if cand_idx not in sampled_indices:
                sampled.append(candidates_sorted[cand_idx])
                sampled_indices.add(cand_idx)
                if len(sampled) == target_size:
                    break

    # 不破坏 ID 顺序（无所谓打乱），routing 算法对输入顺序稳定（详见相关原子卡片）
    return sampled


def _replace_conflicts(
    candidates: list[int],
    current_sampled: list[int],
    units_per_electrode: list[int],
) -> list[int]:
    """冲突替换：把 unit 重复的电极移除，从未抽过且距离最远的 candidates 中补回。

    距离用 electrode_id 之差的平方近似（与 system_api._logical_distance_sq 同思路）。
    """
    if len(units_per_electrode) != len(current_sampled):
        return current_sampled

    unit_to_indices: dict[int, list[int]] = {}
    for index, unit in enumerate(units_per_electrode):
        unit_to_indices.setdefault(unit, []).append(index)

    drop_indices: set[int] = set()
    for unit, indices in unit_to_indices.items():
        if unit < 0 or len(indices) > 1:
            # 失败的（unit=-1）或重复的，全部移除（重复时只保留第一个）
            keep_one = unit >= 0
            for k, idx in enumerate(indices):
                if keep_one and k == 0:
                    continue
                drop_indices.add(idx)

    if not drop_indices:
        return current_sampled

    kept = [el for i, el in enumerate(current_sampled) if i not in drop_indices]
    kept_set = set(kept)
    used_set = set(current_sampled)

    available = [el for el in candidates if el not in used_set]
    need = len(current_sampled) - len(kept)

    # 从 available 中选距离 kept 最远的 need 个
    def min_dist_sq_to_kept(electrode: int) -> int:
        if not kept:
            return 0
        return min((electrode - k) ** 2 for k in kept)

    available_ranked = sorted(available, key=min_dist_sq_to_kept, reverse=True)
    replacements = available_ranked[: max(need, 0)]

    if len(replacements) < need:
        # 候选不足，从剩下随机补
        leftover = [el for el in available if el not in set(replacements)]
        random.shuffle(leftover)
        replacements.extend(leftover[: need - len(replacements)])

    return kept + replacements


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
    """每个 size 的搜索路径：
    1. 第一次：分层抽样得到空间分散的 sampled
    2. 若失败：基于 try_pool 返回的 units 做"冲突替换"——移除 unit 重复或失败的电极，
       从未抽过的 candidates 中拉远距离补回
    3. 替换若不收敛：重新分层抽样
    4. 直至 max_attempts 用尽
    """
    for size in range(target_size, min_size - 1, -1):
        if size > len(candidates):
            print(f"[POOL{pool_index}] size={size} skip (candidates only {len(candidates)})")
            continue
        print(f"\n[POOL{pool_index}] === trying size={size} ===")

        sampled = _stratified_sample(candidates, size)
        last_units: list[int] = []
        for attempt in range(1, max_attempts + 1):
            ok, units, info = _try_pool(record_electrodes, sampled, wells)
            if ok:
                print(f"[POOL{pool_index}] size={size} attempt={attempt} SUCCESS (stratified+replace)")
                return size, sampled, units
            print(f"[POOL{pool_index}] size={size} attempt={attempt} FAIL: {info}")

            # 失败：先尝试冲突替换；若上次也是替换且仍 fail，下次重抽分层样本
            if attempt < max_attempts:
                if last_units == units:
                    # 替换没改善，换一组分层样本
                    sampled = _stratified_sample(candidates, size)
                else:
                    sampled = _replace_conflicts(candidates, sampled, units)
                last_units = units
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
        print(f"[POOL {index}/{len(successes)}] {size} non-conflicting electrodes "
              f"(MODE={MODE})")
        print()
        print(f"  Copy the line below and paste to {target_var} in experiment_config.py:")
        print()
        # json.dumps 强制单行 Python list 字面量，避免终端换行影响复制
        print(f"  {json.dumps(electrodes)}")
        print()
        print(f"  electrode -> stim_unit reference (each row one pair):")
        for electrode, unit in zip(electrodes, units):
            print(f"    {electrode} -> {unit}")
        print("-" * 70)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
