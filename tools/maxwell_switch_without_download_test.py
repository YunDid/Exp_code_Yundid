"""测试：同一 cfg 范围内、改 stim 桥接配置后，是否必须 download 才让硬件生效。

要回答的核心问题：
  「启动期 select(record) + select_stim(A∪B) + route + connect_to_stim(A) + download 之后，
   切换时只调 disconnect(A) + connect(B)+query，**不调 array.download**，
   能否让 B 上真发出刺激？」

设计：4 个阶段，全程录音，每个阶段在 stim 发放前打专属 user_id 的 mx.Event；事后从 .h5
取每个电极通道的波形 + Event frame_number 比对：

  阶段 1 — A 基线（已 download）
    a1 ∈ A，对应 unit_A；mx.send(unit_A.connect(True)) → 发 sequence → 关
    user_id 100 标记 → 期望：在 a1 通道看到刺激波形

  阶段 2 — 切到 B 但**不** download
    array.disconnect_electrode_from_stimulation(a1)
    array.connect_electrode_to_stimulation(b1) + query → unit_B_v1
    （**不**调 array.download）
    mx.send(unit_B_v1.power_up + connect=True + voltage_mode + dac0) → 发 sequence → 关
    user_id 200 标记
    诊断：(i) query 返回的 unit_id 是否有效；(ii) b1 通道是否有波形；(iii) a1 通道是否有波形（应无）

  阶段 3 — 重置后切到 B 并 download（基线对照）
    disconnect_to_stim(b1) + connect_to_stim(a1) + array.download   ← 回到阶段 1 状态
    disconnect_to_stim(a1) + connect_to_stim(b1) + query → unit_B_v2 + array.download
    mx.send(unit_B_v2.connect(True)) → 发 sequence → 关
    user_id 300 标记
    诊断：(i) unit_B_v1 == unit_B_v2 吗；(ii) b1 通道是否有波形（期望有）

  阶段 4 — 关掉所有 stim_unit 输出（不 download）
    mx.send(StimulationUnit(unit).connect(False)) for unit in [unit_A, unit_B_v2]
    发 sequence 看是否还有任何波形
    user_id 400 标记
    诊断：所有通道无波形 → unit 输出层切换有效

判定（事后看 .h5）：
  - 阶段 2 b1 有波形 + 阶段 3 b1 也有波形 → **不需要 download**（lite switch 可进一步去掉 download）
  - 阶段 2 b1 无波形 + 阶段 3 b1 有波形 → **必须 download** （现有实现合理）
  - 阶段 2 a1 有波形 → disconnect 软件命令未真生效 → 必须 download

跑法（Linux 端 venv Python）：
  /home/maxwell/metaboc-env/bin/python tools/maxwell_switch_without_download_test.py

注意：本脚本会真发刺激；不要在生物样本上跑（除非你确认可承受 4 段刺激）。
建议先用空芯片或电阻负载验证。
"""

import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXP_CODE_DIR = THIS_DIR.parent
if str(EXP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_CODE_DIR))

import maxlab as mx

from cfg_utils import extract_electrodes
from experiment_config import CONFIG
from system_api import (
    configure_array_dual_pool,
    initialize_system,
    poweroff_all_stim_units,
    powerup_stim_unit,
)


PULSE_AMPLITUDE_MV = 200
PULSE_PHASE_SAMPLES = 4
N_PULSES_PER_PHASE = 3
# 5s 间隔便于示波器肉眼区分相邻刺激尖刺；
# 单脉冲本身仍是 200μs × 2 双相，间隔与脉冲形状无关。
INTER_PULSE_S = 5.0


def _is_error(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() == "error"


def _query_unit_id(array: mx.Array, electrode: int) -> int | None:
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


def _build_marker_pulse(label: str, user_id: int) -> mx.Sequence:
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    amp_bits = int(PULSE_AMPLITUDE_MV / dac_lsb_mV)
    seq = mx.Sequence()
    seq.append(
        mx.Event(
            0,
            1,
            user_id,
            f"switch_test {label} user_id {user_id} amp_mV {PULSE_AMPLITUDE_MV}",
        )
    )
    seq.append(mx.DAC(0, 512 + amp_bits))
    seq.append(mx.DelaySamples(PULSE_PHASE_SAMPLES))
    seq.append(mx.DAC(0, 512 - amp_bits))
    seq.append(mx.DelaySamples(PULSE_PHASE_SAMPLES))
    seq.append(mx.DAC(0, 512))
    return seq


def _send_n_pulses(label: str, base_user_id: int, n: int) -> None:
    for i in range(n):
        user_id = base_user_id + i
        seq = _build_marker_pulse(f"{label}_pulse{i + 1}", user_id)
        print(f"[SWITCH][{label}] pulse {i + 1}/{n} user_id={user_id}")
        seq.send()
        time.sleep(INTER_PULSE_S)


def main() -> None:
    cfg = CONFIG
    wells = cfg["wells"]
    cfg_path = cfg["config"]

    if not cfg_path:
        raise RuntimeError("CONFIG['config'] is empty; need cfg with prerouted record electrodes.")

    record_electrodes = extract_electrodes(cfg_path)

    # 选两个电极：a1 ∈ 实验组池子，b1 ∈ 对照组池子（都来自 cfg 的 record 范围）
    pool_A = list(cfg["experimental_stim_electrodes"])
    pool_B = list(cfg["control_stim_electrodes"])
    if not pool_A or not pool_B:
        raise RuntimeError("Both experimental_stim_electrodes and control_stim_electrodes must be non-empty.")

    a1 = pool_A[0]
    b1 = pool_B[0]
    print(f"[SWITCH] cfg={cfg_path}")
    print(f"[SWITCH] record electrodes: {len(record_electrodes)}")
    print(f"[SWITCH] a1 (group A first) = {a1}")
    print(f"[SWITCH] b1 (group B first) = {b1}")

    initialize_system()
    mx.activate(wells)

    # 启动期：select(record) + select_stim(A∪B) + route + connect(a1) + download
    array = configure_array_dual_pool(
        electrodes=record_electrodes,
        primary_stim_electrodes=[a1],
        secondary_stim_electrodes=[b1],
        config_file=cfg_path,
    )

    # 阶段 1 准备：连 a1
    if _is_error(array.connect_electrode_to_stimulation(a1)):
        raise RuntimeError(f"Initial connect_electrode_to_stimulation({a1}) failed.")
    unit_A = _query_unit_id(array, a1)
    if unit_A is None:
        raise RuntimeError(f"query_stimulation_at_electrode({a1}) returned no unit.")
    print(f"[SWITCH] phase1 setup: a1={a1} -> unit_A={unit_A}")

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)
    mx.offset()
    time.sleep(3)
    mx.clear_events()

    mx.send(powerup_stim_unit(unit_A))

    saving_cfg = cfg["saving"]
    saving = mx.Saving()
    saving.open_directory(saving_cfg["dir_name"])
    file_basename = f"switch_without_download_{int(time.time())}"
    saving.start_file(file_basename)
    saving.group_define(0, saving_cfg["group_name"], saving_cfg["group_channels"])
    saving.start_recording()
    print(f"[SWITCH] recording started -> {saving_cfg['dir_name']}/{file_basename}.h5")

    diagnostics: dict = {
        "a1": a1,
        "b1": b1,
        "unit_A": unit_A,
    }

    try:
        # ============ 阶段 1：A 基线 ============
        print("\n[SWITCH] === phase 1: baseline on A (downloaded) ===")
        mx.send(mx.StimulationUnit(unit_A).connect(True))
        _send_n_pulses("phase1_A_baseline", base_user_id=100, n=N_PULSES_PER_PHASE)
        mx.send(mx.StimulationUnit(unit_A).connect(False))

        # ============ 阶段 2：切到 B 但不 download ============
        print("\n[SWITCH] === phase 2: switch to B WITHOUT download ===")
        result = array.disconnect_electrode_from_stimulation(a1)
        print(f"[SWITCH] disconnect a1={a1}: {result!r}")
        result = array.connect_electrode_to_stimulation(b1)
        print(f"[SWITCH] connect b1={b1}: {result!r}")
        unit_B_v1 = _query_unit_id(array, b1)
        diagnostics["unit_B_v1_no_download"] = unit_B_v1
        print(f"[SWITCH] phase2 query b1={b1} -> unit_B_v1={unit_B_v1}")

        if unit_B_v1 is None:
            print("[SWITCH] phase2 query returned no unit; will skip phase2 stim attempt.")
        else:
            # 不 download，直接对 query 拿到的 unit 发 stim
            mx.send(powerup_stim_unit(unit_B_v1))
            mx.send(mx.StimulationUnit(unit_B_v1).connect(True))
            _send_n_pulses("phase2_B_no_download", base_user_id=200, n=N_PULSES_PER_PHASE)
            mx.send(mx.StimulationUnit(unit_B_v1).connect(False))

        # ============ 阶段 3：重置后切到 B 并 download（基线对照）============
        print("\n[SWITCH] === phase 3: switch to B WITH download (baseline) ===")
        # 先回到阶段 1 状态
        array.disconnect_electrode_from_stimulation(b1)
        array.connect_electrode_to_stimulation(a1)
        array.download(wells)
        time.sleep(0.5)  # 不做 offset，仅留短暂稳定
        # 再切到 B 并 download
        array.disconnect_electrode_from_stimulation(a1)
        array.connect_electrode_to_stimulation(b1)
        unit_B_v2 = _query_unit_id(array, b1)
        diagnostics["unit_B_v2_with_download"] = unit_B_v2
        print(f"[SWITCH] phase3 query b1={b1} -> unit_B_v2={unit_B_v2}")
        array.download(wells)
        time.sleep(0.5)

        if unit_B_v2 is None:
            print("[SWITCH] phase3 query returned no unit; skip phase3 stim.")
        else:
            mx.send(powerup_stim_unit(unit_B_v2))
            mx.send(mx.StimulationUnit(unit_B_v2).connect(True))
            _send_n_pulses("phase3_B_with_download", base_user_id=300, n=N_PULSES_PER_PHASE)
            mx.send(mx.StimulationUnit(unit_B_v2).connect(False))

        # ============ 阶段 4：关掉 unit 输出，确认无波形 ============
        print("\n[SWITCH] === phase 4: all units off, send sequence as no-op control ===")
        for unit in [unit_A, unit_B_v2 or unit_B_v1]:
            if unit is not None:
                mx.send(mx.StimulationUnit(unit).connect(False))
        _send_n_pulses("phase4_all_off", base_user_id=400, n=N_PULSES_PER_PHASE)

    finally:
        saving.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        saving.stop_file()
        saving.group_delete_all()
        poweroff_all_stim_units()
        print("[SWITCH] recording stopped, .h5 saved")

    print("\n" + "=" * 70)
    print("[SWITCH] diagnostics summary:")
    for k, v in diagnostics.items():
        print(f"  {k} = {v}")
    print("=" * 70)
    print("[SWITCH] manual analysis on .h5:")
    print("  Look at the time-domain waveform on a1 and b1 channels for each phase:")
    print("    phase1 (user_id 100-102): expect waveforms on a1 ONLY")
    print("    phase2 (user_id 200-202): the critical one")
    print("       - waveforms on b1 → switching does NOT need download")
    print("       - no waveforms on b1 (or stuck on a1) → download IS required")
    print("    phase3 (user_id 300-302): expect waveforms on b1 (download confirms baseline)")
    print("    phase4 (user_id 400-402): expect NO waveforms anywhere")
    print()
    print("  Also compare unit_B_v1 (no-download query) vs unit_B_v2 (post-download query):")
    print("    same → query is reliable pre-download for this electrode set")
    print("    different → download changes the unit assignment (need to re-query post-download)")
    print("=" * 70)


if __name__ == "__main__":
    main()
