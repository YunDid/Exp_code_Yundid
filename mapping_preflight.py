import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def _bootstrap_paths() -> None:
    """Make direct script execution work in both dev and MaxLab runtime layouts."""
    this_dir = Path(__file__).resolve().parent
    candidate_paths = [this_dir, this_dir.parent]

    maxlab_root = None
    for parent in this_dir.parents:
        if parent.name == "MaxLab":
            maxlab_root = parent
            break

    if maxlab_root is not None:
        lib_dir = maxlab_root / "python" / "lib"
        if lib_dir.exists():
            candidate_paths.extend(sorted(lib_dir.glob("python*/site-packages")))

    for path in candidate_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


_bootstrap_paths()


try:
    from .experiment_config import CONFIG
    from .recording_api import preview_stim_mapping
    from .protocols import run_mapping_dry_run, save_experiment_json
except ImportError:
    from experiment_config import CONFIG
    from recording_api import preview_stim_mapping
    from protocols import run_mapping_dry_run, save_experiment_json


class _Tee:
    """Write terminal output to both console and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_terminal_to_log(log_path: Path):
    """Mirror stdout/stderr into a log file while preserving terminal output."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_handle:
        sys.stdout = _Tee(original_stdout, log_handle)
        sys.stderr = _Tee(original_stderr, log_handle)
        try:
            print(f"[LOG] terminal output mirrored to: {str(log_path)}")
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _new_run_artifacts(cfg: dict) -> tuple[str, Path]:
    out_dir = Path(cfg["saving"]["dir_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp, out_dir / f"mapping_preflight_{timestamp}.log"


def _build_direct_probe_config(
    base_cfg: dict,
    stim_electrodes: list[int],
) -> dict:
    """直连 probe 配置：strategy=prechecked，不走 neighbor_retry / 不做 substitutions。"""
    cfg = deepcopy(base_cfg)
    mapping_cfg = cfg.setdefault("stim_mapping", {})

    stim_electrodes = list(stim_electrodes)
    max_electrodes = mapping_cfg.get("max_electrodes", 32)
    if len(stim_electrodes) > max_electrodes:
        raise ValueError(
            f"Preflight received {len(stim_electrodes)} electrodes, "
            f"but the configured limit is {max_electrodes}."
        )

    cfg["stim_electrodes"] = stim_electrodes
    mapping_cfg["strategy"] = "prechecked"
    return cfg


def _build_preflight_config(
    base_cfg: dict,
    stim_electrodes: list[int],
    neighbor_radius: int,
    verification_max_iterations: int,
) -> dict:
    cfg = deepcopy(base_cfg)
    mapping_cfg = cfg.setdefault("stim_mapping", {})

    stim_electrodes = list(stim_electrodes)
    max_electrodes = mapping_cfg.get("max_electrodes", 32)
    if len(stim_electrodes) > max_electrodes:
        raise ValueError(
            f"Preflight received {len(stim_electrodes)} electrodes, "
            f"but the configured limit is {max_electrodes}."
        )

    cfg["stim_electrodes"] = stim_electrodes
    mapping_cfg["strategy"] = "neighbor_retry"
    mapping_cfg["neighbor_radius"] = neighbor_radius
    mapping_cfg["verification_max_iterations"] = verification_max_iterations
    return cfg


def _build_final_route_check_config(preflight_cfg: dict, mapping_diag: dict) -> dict:
    final_cfg = deepcopy(preflight_cfg)
    resolved_electrodes = mapping_diag.get("resolved_electrodes", [])
    if not resolved_electrodes:
        raise RuntimeError("Preflight did not produce resolved_electrodes.")

    final_cfg["stim_electrodes"] = resolved_electrodes
    final_cfg["stim_mapping"]["strategy"] = "prechecked"
    return final_cfg


def _run_final_route_check(preflight_cfg: dict, mapping_diag: dict) -> dict:
    """Verify resolved_electrodes with the same direct route used by formal runs."""
    final_cfg = _build_final_route_check_config(preflight_cfg, mapping_diag)

    print("[PRECHECK] final direct route check starts.")
    print(f"[PRECHECK] final route electrodes={final_cfg['stim_electrodes']}")
    final_diag = preview_stim_mapping(final_cfg)

    conflicts = final_diag.get("conflicts", [])
    if conflicts:
        print("[PRECHECK] final direct route still has conflicts:")
        for conflict in conflicts:
            print(
                f"[PRECHECK] stim_unit {conflict['stim_unit']} conflict: "
                f"electrodes {conflict['electrodes']}"
            )
    else:
        stim_units = [
            final_diag.get("electrode2unit", {}).get(str(electrode), "?")
            for electrode in final_cfg["stim_electrodes"]
        ]
        print("[PRECHECK] final direct route check passed.")
        print(f"[PRECHECK] final stimulation units={stim_units}")

    return final_diag


def _override_mapping_diag_with_narrow_set(
    mapping_diag: dict, final_route_diag: dict
) -> None:
    """Override mapping_diag's unit-level fields with narrow-set routing results.

    The neighbor-retry phase queries unit ids while the array is routed with the
    expanded candidate pool. The formal experiment uses only the resolved subset,
    which is a different routing input set; query results may differ. Use the
    narrow-set query (final_route_check) as the authority for what the formal
    experiment will actually see.
    """
    expanded_e2u = mapping_diag.get("electrode2unit", {})
    narrow_e2u = final_route_diag.get("electrode2unit", {})
    diffs = []
    for el_str, narrow_unit in narrow_e2u.items():
        expanded_unit = expanded_e2u.get(el_str)
        if expanded_unit is not None and expanded_unit != narrow_unit:
            diffs.append((el_str, expanded_unit, narrow_unit))

    if diffs:
        print("[PRECHECK] routing-set difference detected (expanded routing vs narrow routing):")
        for el, e_unit, n_unit in diffs:
            print(f"[PRECHECK]   electrode {el}: expanded={e_unit}, narrow={n_unit}")
        print(
            "[PRECHECK] mapping_diag unit fields overridden by narrow-set routing "
            "(formal experiment will use these)."
        )
    else:
        print("[PRECHECK] expanded vs narrow routing produces identical unit map; no override needed.")

    for key in (
        "n_electrodes",
        "n_units_unique",
        "n_conflict_units",
        "extra_electrodes_due_to_conflicts",
        "conflicts",
        "unit2electrodes",
        "electrode2unit",
        "resolved_electrode2unit",
    ):
        if key in final_route_diag:
            mapping_diag[key] = final_route_diag[key]


def _print_electrode_unit_pairs(
    electrodes: list[int], electrode2unit: dict
) -> None:
    """每行打印一对 electrode -> stim_unit，便于交叉校验。"""
    print("[PRECHECK] electrode -> stim_unit:")
    for electrode in electrodes:
        unit = electrode2unit.get(str(electrode), "?")
        print(f"  {electrode} -> {unit}")


def main() -> None:
    requested_stim_electrodes = [
        304, 1019, 2024, 2749, 3838, 7346, 5984, 6076,
        7963, 10482, 8461, 10506, 18617, 12312, 13166, 14097,
        12630, 16489, 17444, 13755, 13369, 18462, 18653, 19678,
        20524, 20717, 20506, 22772, 23035, 23204, 24739, 25563,
    ]
    neighbor_radius = 50
    verification_max_iterations = 100

    timestamp, log_path = _new_run_artifacts(CONFIG)

    with _tee_terminal_to_log(log_path):
        print(
            f"[PRECHECK] requested electrodes ({len(requested_stim_electrodes)}): "
            f"{requested_stim_electrodes}"
        )

        # ===== 阶段 1：直连 probe =====
        # 不走 neighbor_retry / 不替换电极；只看输入电极本身是否在 select+route 后无冲突。
        # 无冲突就直接结束；有冲突才进入阶段 2。
        print("\n[PRECHECK] step 1: probe direct mapping (no neighbor search) ...")
        try:
            direct_cfg = _build_direct_probe_config(CONFIG, requested_stim_electrodes)
            direct_diag = preview_stim_mapping(direct_cfg)
        except Exception as exc:
            print(f"[PRECHECK] direct probe failed: {exc}")
            print("[PRECHECK] do not start the formal experiment.")
            return

        if not direct_diag.get("conflicts"):
            print("[PRECHECK] direct mapping has no conflicts; ready to import as-is.")
            _print_electrode_unit_pairs(
                requested_stim_electrodes, direct_diag.get("electrode2unit", {})
            )

            save_experiment_json(
                cfg=direct_cfg,
                out_name_prefix="mapping_preflight",
                extra_meta={
                    "run_type": "mapping_preflight",
                    "mapping_preflight": {
                        "phase": "direct_probe_only",
                        "requested_electrodes": list(requested_stim_electrodes),
                        "resolved_electrodes": list(requested_stim_electrodes),
                        "n_substitutions": 0,
                        "n_conflict_units": 0,
                        "electrode2unit": direct_diag.get("electrode2unit", {}),
                        "unit2electrodes": direct_diag.get("unit2electrodes", {}),
                    },
                },
                timestamp=timestamp,
            )
            return

        # ===== 阶段 2：有冲突 → neighbor_retry 邻居替换 + final direct route check =====
        n_conflict = direct_diag.get("n_conflict_units", 0)
        print(
            f"[PRECHECK] direct mapping has {n_conflict} conflict unit(s); "
            f"start neighbor_retry search."
        )
        print(f"[PRECHECK] neighbor_radius={neighbor_radius}")
        print(f"[PRECHECK] verification_max_iterations={verification_max_iterations}")

        cfg = _build_preflight_config(
            CONFIG,
            stim_electrodes=requested_stim_electrodes,
            neighbor_radius=neighbor_radius,
            verification_max_iterations=verification_max_iterations,
        )
        mapping_diag = run_mapping_dry_run(cfg)
        final_route_diag = None
        final_route_error = None

        mapping_unresolved = mapping_diag.get("unresolved_requested_electrodes", [])
        if mapping_diag.get("conflicts") or mapping_unresolved:
            print("[PRECHECK] skip final direct route check because neighbor retry did not produce a clean mapping.")
        else:
            try:
                final_route_diag = _run_final_route_check(cfg, mapping_diag)
            except Exception as exc:
                final_route_error = str(exc)
                print(f"[PRECHECK] final direct route check failed: {final_route_error}")

            if final_route_diag is not None and not final_route_diag.get("conflicts"):
                _override_mapping_diag_with_narrow_set(mapping_diag, final_route_diag)

        save_experiment_json(
            cfg=cfg,
            out_name_prefix="mapping_preflight",
            extra_meta={
                "run_type": "mapping_preflight",
                "mapping_preflight": mapping_diag,
                "final_route_check": final_route_diag,
                "final_route_error": final_route_error,
            },
            timestamp=timestamp,
        )

        if final_route_error is not None:
            print("[PRECHECK] do not start the formal experiment.")
        elif final_route_diag and final_route_diag.get("conflicts"):
            print("[PRECHECK] final direct route is still conflicted; do not start the formal experiment.")
        elif mapping_diag.get("conflicts"):
            print("[PRECHECK] unresolved conflicts remain; do not start the formal experiment.")
        elif mapping_unresolved:
            print(f"[PRECHECK] unresolved electrodes remain: {mapping_unresolved}")
            print("[PRECHECK] do not start the formal experiment.")
        elif final_route_diag is None:
            print("[PRECHECK] final direct route check did not run; do not start the formal experiment.")
        else:
            print("[PRECHECK] resolved_electrodes is the electrode group to import.")
            print(f"[PRECHECK] resolved_electrodes={mapping_diag.get('resolved_electrodes', [])}")


if __name__ == "__main__":
    main()
