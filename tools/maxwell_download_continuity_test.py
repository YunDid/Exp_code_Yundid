"""测试 array.download(wells) 是否打断 .h5 录音 / stim Event 连续性。

要回答的核心问题：
  「在录音过程中调一次 array.download(wells)，前后已写入的 mx.Event 标记是否仍连续？
   是否有 Event 丢失？录音 frame 是否中断？」

设计：
  启动一次完整 routing → 启动录音 → 第 1 段每 1s 发 1 个带唯一 user_id 的 stim Event
  共 5 个 → 调 array.download(wells)（不修改 array 配置，纯重发同一份）→ 第 2 段
  每 1s 发 1 个 stim Event 共 5 个 → 停止录音。

判定（事后看 .h5 frame metadata）：
  - 10 个 Event 全部存在（user_id 100-104 + 200-204）→ download 不丢 Event
  - 第 1 段内部 frame_number 间隔 ≈ 1s × 20000 samples
  - 第 2 段内部 frame_number 间隔 ≈ 同上
  - 跨 download 间隔（seg1[-1] → seg2[0]）≈ 上述间隔 + download 调用耗时 × 20 samples/ms
    与其他间隔差异 < 5% → download 不打断录音
  - 否则报告 download 的具体扰动幅度

跑法（Linux 端，必须用 venv Python）：
  /home/maxwell/metaboc-env/bin/python tools/maxwell_download_continuity_test.py

本脚本不修改 array 配置，仅在录音中插入一次空 download 重发，安全反复跑。
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
    configure_and_powerup_stim_units,
    configure_array,
    connect_stim_units_to_stim_electrodes,
    initialize_system,
    poweroff_all_stim_units,
)


N_EVENTS_PER_SEGMENT = 5
INTERVAL_S = 1.0
PULSE_AMPLITUDE_MV = 200
PULSE_PHASE_SAMPLES = 4


def _build_marker_pulse(label: str, user_id: int) -> mx.Sequence:
    """带唯一 user_id 标记的最小双相脉冲。Event 与 stim 同 sequence 共享 frame_number。"""
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    amp_bits = int(PULSE_AMPLITUDE_MV / dac_lsb_mV)

    seq = mx.Sequence()
    seq.append(
        mx.Event(
            0,
            1,
            user_id,
            f"continuity_test {label} user_id {user_id} amp_mV {PULSE_AMPLITUDE_MV}",
        )
    )
    seq.append(mx.DAC(0, 512 + amp_bits))
    seq.append(mx.DelaySamples(PULSE_PHASE_SAMPLES))
    seq.append(mx.DAC(0, 512 - amp_bits))
    seq.append(mx.DelaySamples(PULSE_PHASE_SAMPLES))
    seq.append(mx.DAC(0, 512))
    return seq


def main() -> None:
    cfg = CONFIG
    wells = cfg["wells"]
    cfg_path = cfg["config"]

    if not cfg_path:
        raise RuntimeError(
            "CONFIG['config'] is empty; download continuity test requires a cfg "
            "with prerouted recording electrodes."
        )

    record_electrodes = extract_electrodes(cfg_path)
    # 只用 1 个 stim 电极发 marker pulse，最小化对实验生物样本的扰动
    stim_electrodes = list(cfg["experimental_stim_electrodes"][:1])

    print(f"[CONT] cfg={cfg_path}")
    print(f"[CONT] record electrodes from cfg: {len(record_electrodes)}")
    print(f"[CONT] marker stim electrode: {stim_electrodes}")
    print(
        f"[CONT] plan: 2 segments × {N_EVENTS_PER_SEGMENT} events,"
        f" interval={INTERVAL_S}s, with array.download(wells) between segments"
    )

    initialize_system()
    mx.activate(wells)

    array = configure_array(record_electrodes, stim_electrodes)
    stim_units = connect_stim_units_to_stim_electrodes(stim_electrodes, array)
    print(f"[CONT] stim_units after connect+query: {stim_units}")

    array.download(wells)
    time.sleep(mx.Timing.waitAfterDownload)

    mx.offset()
    time.sleep(3)
    mx.clear_events()

    configure_and_powerup_stim_units(stim_units)

    saving_cfg = cfg["saving"]
    saving = mx.Saving()
    saving.open_directory(saving_cfg["dir_name"])
    file_basename = f"download_continuity_{int(time.time())}"
    saving.start_file(file_basename)
    saving.group_define(0, saving_cfg["group_name"], saving_cfg["group_channels"])
    saving.start_recording()
    print(f"[CONT] recording started -> {saving_cfg['dir_name']}/{file_basename}.h5")

    download_durations_s: list[float] = []

    try:
        # connect=True 让 stim_unit 输出对外接通，发 sequence 时电极有真实波形
        for unit in stim_units:
            mx.send(mx.StimulationUnit(unit).connect(True))

        # ===== 第 1 段：5 个 Event =====
        seg1_user_ids = list(range(100, 100 + N_EVENTS_PER_SEGMENT))
        seg1_send_walltime: list[float] = []
        for index, user_id in enumerate(seg1_user_ids):
            t_send = time.time()
            seq = _build_marker_pulse(f"seg1_idx{index}", user_id)
            seq.send()
            seg1_send_walltime.append(t_send)
            print(f"[CONT][SEG1] event #{index + 1}/{N_EVENTS_PER_SEGMENT} user_id={user_id} sent at t={t_send:.3f}")
            time.sleep(INTERVAL_S)

        # ===== 中间：纯 download，不修改 array 配置 =====
        print("[CONT] >>> calling array.download(wells) — no config change, pure idempotent push")
        t_dl0 = time.time()
        array.download(wells)
        t_dl1 = time.time()
        download_durations_s.append(t_dl1 - t_dl0)
        print(f"[CONT] >>> download returned in {(t_dl1 - t_dl0) * 1000:.1f}ms")

        # ===== 第 2 段：5 个 Event =====
        seg2_user_ids = list(range(200, 200 + N_EVENTS_PER_SEGMENT))
        seg2_send_walltime: list[float] = []
        for index, user_id in enumerate(seg2_user_ids):
            t_send = time.time()
            seq = _build_marker_pulse(f"seg2_idx{index}", user_id)
            seq.send()
            seg2_send_walltime.append(t_send)
            print(f"[CONT][SEG2] event #{index + 1}/{N_EVENTS_PER_SEGMENT} user_id={user_id} sent at t={t_send:.3f}")
            time.sleep(INTERVAL_S)

        # 关掉 unit 输出
        for unit in stim_units:
            mx.send(mx.StimulationUnit(unit).connect(False))

        print(f"[CONT] all events sent — seg1 user_ids={seg1_user_ids}  seg2 user_ids={seg2_user_ids}")

        # 走时计算（对照 frame_number gap 用）
        if seg1_send_walltime and seg2_send_walltime:
            cross_walltime_gap = seg2_send_walltime[0] - seg1_send_walltime[-1]
            print(f"[CONT] walltime gap seg1[-1] -> seg2[0] = {cross_walltime_gap:.3f}s "
                  f"(includes INTERVAL_S {INTERVAL_S}s + download {download_durations_s[0]*1000:.1f}ms)")
            print(f"[CONT] expected frame_number gap ≈ {cross_walltime_gap * 20000:.0f} samples (assuming 20 kHz)")

    finally:
        saving.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        saving.stop_file()
        saving.group_delete_all()
        poweroff_all_stim_units()
        print("[CONT] recording stopped, .h5 saved")

    print("\n" + "=" * 70)
    print("[CONT] manual analysis on .h5:")
    print("  1. Open the h5 file (e.g. via Maxwell GUI or h5py).")
    print("  2. List all Events written to frame metadata:")
    print(f"       expect 10 events with user_id ∈ {{{seg1_user_ids[0]}..{seg1_user_ids[-1]},"
          f" {seg2_user_ids[0]}..{seg2_user_ids[-1]}}}")
    print("  3. Compute each Event's frame_number and adjacent gaps:")
    print(f"       - seg1 internal gaps (4 gaps): expect ≈ {INTERVAL_S * 20000:.0f} samples each")
    print(f"       - seg2 internal gaps (4 gaps): expect ≈ {INTERVAL_S * 20000:.0f} samples each")
    print(f"       - cross gap seg1[-1] -> seg2[0]: expect ≈ {INTERVAL_S * 20000 + download_durations_s[0]*20000 if download_durations_s else 0:.0f} samples")
    print("  4. Verdict:")
    print("       - all 10 events present and gaps within 5% of expected → download is non-disruptive")
    print("       - any event missing or cross gap >> internal gaps → download disrupts recording")
    print("=" * 70)


if __name__ == "__main__":
    main()
