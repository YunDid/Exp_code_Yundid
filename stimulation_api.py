import random
import time
from typing import List, Optional

import maxlab as mx


event_counter = 1


def create_stim_pulse(
    seq: mx.Sequence, amplitude: int, delay_samples: int, amplitude_mV: int
) -> mx.Sequence:
    """Append one biphasic pulse to a sequence."""
    global event_counter

    event_counter += 1
    seq.append(
        mx.Event(
            0,
            1,
            event_counter,
            f"amplitude {amplitude_mV} event_id {event_counter}",
        )
    )
    seq.append(mx.DAC(0, 512 + amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512 - amplitude))
    seq.append(mx.DelaySamples(delay_samples))
    seq.append(mx.DAC(0, 512))
    return seq


def prepare_stim_sequence(
    number_pulses_per_train: int,
    inter_pulse_interval: int,
    phase: int,
    amplitude: int,
    changing_amplitude: Optional[bool] = False,
    max_amplitude: Optional[int] = None,
    amplitude_interval: Optional[int] = None,
) -> mx.Sequence:
    """Prepare a pulse-train sequence."""
    seq = mx.Sequence()
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())

    if changing_amplitude:
        if max_amplitude is None or amplitude_interval is None:
            raise ValueError(
                "Both max_amplitude and amplitude_interval are required for changing_amplitude."
            )

        for cur_amplitude in range(amplitude, max_amplitude, amplitude_interval):
            for _ in range(number_pulses_per_train):
                seq = create_stim_pulse(
                    seq,
                    int(cur_amplitude / dac_lsb_mV),
                    phase,
                    cur_amplitude,
                )
                seq.append(mx.DelaySamples(inter_pulse_interval))
            seq.append(mx.DelaySamples(inter_pulse_interval))
    else:
        for _ in range(number_pulses_per_train):
            seq = create_stim_pulse(
                seq,
                int(amplitude / dac_lsb_mV),
                phase,
                amplitude,
            )
            seq.append(mx.DelaySamples(inter_pulse_interval))

    return seq


def send_stim_pulses_all_units(seq: mx.Sequence, number_pulse_trains: int) -> None:
    """Send the same pulse train to all connected units."""
    for _ in range(number_pulse_trains):
        print("Send pulse")
        seq.send()
        time.sleep(10)


def send_stim_pulses_units_sequentially(
    seq: mx.Sequence, stim_units: List[int]
) -> None:
    """Send the same pulse train to the given units one-by-one."""
    for stim_unit in stim_units:
        print(f"Power up stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit)
        stim.power_up(True).connect(True).set_voltage_mode().dac_source(0)
        mx.send(stim)
        print("Send pulse")
        seq.send()
        print(f"Power down stimulation unit {stim_unit}")
        stim = mx.StimulationUnit(stim_unit).power_up(False)
        mx.send(stim)
        time.sleep(2)


def build_mapping_diag_from_el2unit(el2unit: dict[int, int]) -> dict:
    """Build a JSON-safe diagnostic summary for electrode->unit conflicts."""
    unit2els: dict[int, list[int]] = {}
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


def merge_mapping_diagnostics(
    existing_diag: Optional[dict],
    el2unit: dict[int, int],
) -> dict:
    """Preserve setup-time diagnostics while refreshing conflict statistics."""
    merged = dict(existing_diag or {})
    merged.update(build_mapping_diag_from_el2unit(el2unit))
    return merged


def stimulate_units_random_order(
    seq: mx.Sequence,
    stim_units: List[int],
    stim_electrodes: List[int],
    repeats: int = 5,
    sleep_between_units_s: float = 10.0,
    cfg: dict | None = None,
) -> List[List[int]]:
    """Stimulate one unit at a time in a randomized electrode order."""
    el2unit = {el: unit for el, unit in zip(stim_electrodes, stim_units)}

    if cfg is not None:
        cfg["stim_mapping_diagnostics"] = merge_mapping_diagnostics(
            cfg.get("stim_mapping_diagnostics"),
            el2unit,
        )
        print(
            "[MAPPING]",
            cfg["stim_mapping_diagnostics"]["n_conflict_units"],
            "conflict units; extra_electrodes_due_to_conflicts=",
            cfg["stim_mapping_diagnostics"]["extra_electrodes_due_to_conflicts"],
        )

    all_orders: List[List[int]] = []

    for unit in stim_units:
        mx.send(mx.StimulationUnit(unit).connect(False))

    for _ in range(repeats):
        order = stim_electrodes.copy()
        random.shuffle(order)
        all_orders.append(order)

        for electrode in order:
            unit = el2unit[electrode]
            resolved_electrode = electrode
            if cfg is not None:
                resolved_electrode = cfg["stim_mapping_diagnostics"].get(
                    "requested_to_resolved",
                    {},
                ).get(str(electrode), electrode)

            for current_unit in stim_units:
                mx.send(mx.StimulationUnit(current_unit).connect(False))

            mx.send(mx.StimulationUnit(unit).connect(True))

            if resolved_electrode != electrode:
                print(
                    f"Stimulate requested electrode {electrode} via resolved electrode "
                    f"{resolved_electrode} (stim_unit {unit})"
                )
            else:
                print(f"Stimulate electrode {electrode} (stim_unit {unit})")
            seq.send()
            time.sleep(sleep_between_units_s)

    for unit in stim_units:
        mx.send(mx.StimulationUnit(unit).connect(False))

    return all_orders


def build_sequence_from_cfg(cfg: dict) -> mx.Sequence:
    """Build the random-stimulation pulse train from cfg['random_stim'].""" 
    stim_cfg = cfg["random_stim"]
    return prepare_stim_sequence(
        number_pulses_per_train=stim_cfg["pulses_per_electrode"],
        inter_pulse_interval=stim_cfg["inter_pulse_interval"],
        phase=stim_cfg["phase"],
        amplitude=stim_cfg["amplitude_mV"],
    )


def disconnect_all_units(stim_units_all: List[int]) -> None:
    """Disconnect all stimulation units."""
    for unit in stim_units_all:
        mx.send(mx.StimulationUnit(unit).connect(False))
    time.sleep(1)


def connect_units_subset(stim_units_all: List[int], subset: List[int]) -> None:
    """Disconnect all units, then connect only the requested subset."""
    disconnect_all_units(stim_units_all)
    for unit in subset:
        mx.send(mx.StimulationUnit(unit).connect(True))


def build_single_pulse_sequence(
    cfg: dict, label: str, pulse_config_key: str = "test_pulse"
) -> mx.Sequence:
    """Build one pulse with one event label for protocol blocks."""
    global event_counter

    stim_cfg = cfg[pulse_config_key]
    dac_lsb_mV = float(mx.query_DAC_lsb_mV())
    amp_bits = int(stim_cfg["amplitude_mV"] / dac_lsb_mV)
    phase = stim_cfg["phase"]
    dac_channel = stim_cfg["dac_channel"]

    seq = mx.Sequence()

    event_counter += 1
    seq.append(
        mx.Event(
            0,
            1,
            event_counter,
            f"label {label} event_id {event_counter} amp_mV {stim_cfg['amplitude_mV']}",
        )
    )
    seq.append(mx.DAC(dac_channel, 512 + amp_bits))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac_channel, 512 - amp_bits))
    seq.append(mx.DelaySamples(phase))
    seq.append(mx.DAC(dac_channel, 512))

    return seq
