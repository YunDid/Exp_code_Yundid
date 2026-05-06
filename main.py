import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def _bootstrap_paths() -> None:
    """Make direct script execution work in both dev and MaxLab runtime layouts."""
    this_dir = Path(__file__).resolve().parent
    candidate_paths = [this_dir, this_dir.parent]

    # On the experiment machine, `maxlab` is installed under:
    #   MaxLab/python/lib/pythonX.Y/site-packages
    # while this script lives under:
    #   MaxLab/share/python/Exp_code
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
    from .protocols import (
        run_protocol,
        run_protocol_control,
        run_random_stim_experiment,
        save_experiment_json,
    )
except ImportError:
    from experiment_config import CONFIG
    from protocols import (
        run_protocol,
        run_protocol_control,
        run_random_stim_experiment,
        save_experiment_json,
    )


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


def _new_run_artifacts(out_name_prefix: str) -> tuple[str, Path]:
    """Return one timestamp and its paired terminal log path."""
    out_dir = Path(CONFIG["saving"]["dir_name"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{out_name_prefix}_{timestamp}.log"
    return timestamp, log_path


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


def main() -> None:
    # 正式实验严格走 strategy=prechecked 直连路径：
    #   - mapping_preflight.py 跑完后产出无冲突的 resolved_electrodes 列表；
    #   - 用户把 resolved_electrodes 填回 experiment_config.py 的 stim_electrodes；
    #   - 正式实验只在此处校验「prechecked」，再下沉到 setup_routing_and_units 直接
    #     connect+query+download，不再触发任何邻居搜索 / 替换 / 验证迭代。
    # 若仍出现冲突，说明输入电极组本身有问题，应回到 mapping_preflight 重选。
    if CONFIG.get("stim_mapping", {}).get("strategy") == "neighbor_retry":
        raise RuntimeError(
            "neighbor_retry is preflight-only. Run mapping_preflight.py first, then "
            "put the resolved electrodes into experiment_config.py for the formal experiment."
        )

    # 1. Random stimulation experiment
    # timestamp, log_path = _new_run_artifacts("random")
    # with _tee_terminal_to_log(log_path):
    #     all_orders = run_random_stim_experiment(CONFIG)
    #     print("Random orders:", all_orders)
    #     save_experiment_json(
    #         cfg=CONFIG,
    #         out_name_prefix="random",
    #         random_orders=all_orders,
    #         timestamp=timestamp,
    #     )

    # 2. Test/train protocol experiment
    # 按 CONFIG["experiment_group"] 分流：
    #   "experimental"：原 run_protocol，路径与逻辑完全不变。
    #   "control"：run_protocol_control，对照组每 cycle 切换 routing 跑随机 10 unit 训练。
    experiment_group = CONFIG.get("experiment_group", "experimental")
    if experiment_group not in ("experimental", "control"):
        raise ValueError(
            f"Unsupported experiment_group: {experiment_group!r}. "
            "Expected 'experimental' or 'control'."
        )

    out_name_prefix = "protocol" if experiment_group == "experimental" else "protocol_control"
    timestamp, log_path = _new_run_artifacts(out_name_prefix)
    with _tee_terminal_to_log(log_path):
        print(f"[MAIN] experiment_group={experiment_group}")
        if experiment_group == "experimental":
            results = run_protocol(CONFIG)
        else:
            results = run_protocol_control(CONFIG)
        print("Protocol results:", results)
        save_experiment_json(
            cfg=CONFIG,
            out_name_prefix=out_name_prefix,
            protocol_results=results,
            timestamp=timestamp,
        )


if __name__ == "__main__":
    main()
