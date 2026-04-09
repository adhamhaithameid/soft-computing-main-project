#!/usr/bin/env python3
"""Interactive cross-platform launcher for the full soft-computing workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.paths import (  # noqa: E402
    RESULTS_HISTORY_INDEX_JSON,
    RESULTS_HISTORY_INDEX_MD,
    RESULTS_HISTORY_RUNS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_REPORTS_DIR,
    ensure_structure,
)


@dataclass
class RunContext:
    run_number: int
    run_id: str
    started_utc: str
    run_dir: Path
    mode: str
    platform_profile: str
    host_platform: str


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _detect_host_profile() -> str:
    value = platform.system().lower()
    if value.startswith("win"):
        return "windows"
    if value == "darwin":
        return "mac"
    return "linux"


def _missing_required_modules() -> List[str]:
    required = ["numpy", "pandas", "sklearn", "scipy"]
    return [name for name in required if importlib.util.find_spec(name) is None]


def _maybe_reexec_in_project_venv() -> None:
    """Relaunch using project venv when current interpreter lacks required deps."""
    if os.environ.get("SC_RUNALL_BOOTSTRAPPED") == "1":
        return

    missing = _missing_required_modules()
    if not missing:
        return

    candidates = [
        ROOT / ".venv311" / "bin" / "python",
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv311" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if Path(sys.executable).resolve() == candidate.resolve():
            continue

        print(
            "Current Python environment is missing project dependencies "
            f"({', '.join(missing)})."
        , flush=True)
        print(f"Relaunching with project interpreter: {candidate}", flush=True)
        env = os.environ.copy()
        env["SC_RUNALL_BOOTSTRAPPED"] = "1"
        code = subprocess.call([str(candidate), *sys.argv], cwd=ROOT, env=env)
        raise SystemExit(code)


def _prompt_choice(title: str, options: List[str], default: str) -> str:
    print(f"\n{title}")
    for idx, option in enumerate(options, start=1):
        marker = " (default)" if option == default else ""
        print(f"  {idx}. {option}{marker}")
    raw = input("> Select option number and press Enter: ").strip()
    if not raw:
        return default
    try:
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index]
    except ValueError:
        pass
    print(f"Invalid selection. Using default: {default}")
    return default


def _load_history() -> List[Dict[str, object]]:
    if not RESULTS_HISTORY_INDEX_JSON.exists():
        return []
    try:
        payload = json.loads(RESULTS_HISTORY_INDEX_JSON.read_text(encoding="utf-8"))
        runs = payload.get("runs", [])
        if isinstance(runs, list):
            return runs
    except Exception:
        pass
    return []


def _save_history(runs: List[Dict[str, object]]) -> None:
    RESULTS_HISTORY_INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_HISTORY_INDEX_JSON.write_text(
        json.dumps({"runs": runs}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Run History",
        "",
        "Each invocation of `python run_all.py` creates one archived run entry.",
        "",
        "| Run | Timestamp (UTC) | Mode | Platform Profile | Runtime (HH:MM:SS) | Validation | Folder |",
        "|---:|:---|:---|:---|:---:|:---:|:---|",
    ]
    for row in runs:
        lines.append(
            "| {run_number} | {started_utc} | {mode} | {platform_profile} | {runtime_hms} | {validation} | `{run_id}` |".format(
                run_number=row.get("run_number", "?"),
                started_utc=row.get("started_utc", "N/A"),
                mode=row.get("mode", "N/A"),
                platform_profile=row.get("platform_profile", "N/A"),
                runtime_hms=row.get("runtime_hms", "N/A"),
                validation=row.get("validation", "N/A"),
                run_id=row.get("run_id", "N/A"),
            )
        )
    RESULTS_HISTORY_INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_run_context(mode: str, platform_profile: str) -> RunContext:
    history = _load_history()
    run_number = len(history) + 1
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run{run_number}_{stamp}"
    run_dir = RESULTS_HISTORY_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_number=run_number,
        run_id=run_id,
        started_utc=started.isoformat(),
        run_dir=run_dir,
        mode=mode,
        platform_profile=platform_profile,
        host_platform=platform.platform(),
    )


def _run_step(label: str, cmd: List[str], env: Dict[str, str] | None = None) -> float:
    print(f"\n==> {label}")
    print(" ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    duration = time.perf_counter() - start
    print(f"[done] {label} in {_format_hms(duration)}")
    return duration


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _extract_validation_status(report_text: str) -> str:
    for line in report_text.splitlines():
        if line.lower().startswith("validation result:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


def _archive_run_outputs(ctx: RunContext, step_times: Dict[str, float]) -> Tuple[dict, str]:
    manifest_path = RESULTS_METRICS_DIR / "cartesian_run_manifest.json"
    validation_path = RESULTS_REPORTS_DIR / "cartesian_validation_report.md"
    comparison_path = RESULTS_REPORTS_DIR / "cartesian_comparison_report.md"

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validation_text = ""
    if validation_path.exists():
        validation_text = validation_path.read_text(encoding="utf-8")
    validation_status = _extract_validation_status(validation_text) if validation_text else "UNKNOWN"

    # Keep run-level immutable snapshots for later professor review.
    _copy_if_exists(manifest_path, ctx.run_dir / "cartesian_run_manifest.json")
    _copy_if_exists(validation_path, ctx.run_dir / "cartesian_validation_report.md")
    _copy_if_exists(comparison_path, ctx.run_dir / "cartesian_comparison_report.md")

    total_runtime = sum(step_times.values())
    run_report = [
        f"# Run {ctx.run_number} Summary",
        "",
        f"- Run id: `{ctx.run_id}`",
        f"- Timestamp (UTC): {ctx.started_utc}",
        f"- Mode: `{ctx.mode}`",
        f"- Platform profile: `{ctx.platform_profile}`",
        f"- Host platform: `{ctx.host_platform}`",
        f"- Total runtime (sec): {total_runtime:.2f}",
        f"- Total runtime (HH:MM:SS): {_format_hms(total_runtime)}",
        f"- Validation status: `{validation_status}`",
        "",
        "## Step Runtime Breakdown",
        "",
    ]
    for step, sec in step_times.items():
        run_report.append(f"- {step}: {sec:.2f} sec ({_format_hms(sec)})")

    if manifest:
        run_report.extend(
            [
                "",
                "## Manifest Snapshot",
                "",
                f"- Expected combos: {manifest.get('expected_combos', 'N/A')}",
                f"- Expected fold evals: {manifest.get('expected_fold_evals', 'N/A')}",
                f"- Rows written: {manifest.get('rows_written', 'N/A')}",
                f"- Completed ok: {manifest.get('completed_ok', 'N/A')}",
                f"- Skipped or failed: {manifest.get('skipped_or_failed', 'N/A')}",
                f"- Runtime (sec): {manifest.get('runtime_sec', 'N/A')}",
                f"- Execution device: {manifest.get('execution_device', 'N/A')}",
                f"- Acceleration backend: {manifest.get('acceleration_backend', 'N/A')}",
            ]
        )

    (ctx.run_dir / "run_summary.md").write_text("\n".join(run_report) + "\n", encoding="utf-8")
    return manifest, validation_status


def _update_run_history(
    ctx: RunContext,
    step_times: Dict[str, float],
    validation_status: str,
    manifest: dict,
) -> None:
    runs = _load_history()
    total_runtime = sum(step_times.values())
    runs.append(
        {
            "run_number": ctx.run_number,
            "run_id": ctx.run_id,
            "started_utc": ctx.started_utc,
            "mode": ctx.mode,
            "platform_profile": ctx.platform_profile,
            "host_platform": ctx.host_platform,
            "runtime_sec": round(total_runtime, 3),
            "runtime_hms": _format_hms(total_runtime),
            "validation": validation_status,
            "execution_device": manifest.get("execution_device", "unknown"),
            "acceleration_backend": manifest.get("acceleration_backend", "unknown"),
            "completed_ok": manifest.get("completed_ok", "unknown"),
            "skipped_or_failed": manifest.get("skipped_or_failed", "unknown"),
            "expected_fold_evals": manifest.get("expected_fold_evals", "unknown"),
        }
    )
    _save_history(runs)


def _print_validation_block(
    manifest: dict,
    validation_status: str,
    allow_partial: bool,
    launcher_runtime_sec: float,
) -> None:
    expected_combos = manifest.get("expected_combos", "N/A")
    expected_evals = manifest.get("expected_fold_evals", "N/A")
    rows_written = manifest.get("rows_written", "N/A")
    completed_ok = manifest.get("completed_ok", "N/A")
    skipped = manifest.get("skipped_or_failed", "N/A")
    runtime_sec = float(manifest.get("runtime_sec", 0.0) or 0.0)
    figure_count = len(list((ROOT / "results" / "figures").glob("cartesian_*.png")))
    mode_label = "partial allowed" if allow_partial else "strict full run"
    timestamp = datetime.now(timezone.utc).isoformat()

    print("\n# Cartesian Validation Report")
    print()
    print(f"- Timestamp (UTC): {timestamp}")
    print(f"- Expected combos: {expected_combos}")
    print(f"- Expected fold evals: {expected_evals}")
    print(f"- Metrics rows: {rows_written}")
    print(f"- Completed ok: {completed_ok}")
    print(f"- Skipped or failed: {skipped}")
    print(f"- Figures found: {figure_count}")
    print(f"- Mode: {mode_label}")
    print(f"- Benchmark runtime (sec): {runtime_sec:.2f}")
    print(f"- Benchmark runtime (HH:MM:SS): {_format_hms(runtime_sec)}")
    print(f"- Launcher runtime (sec): {launcher_runtime_sec:.2f}")
    print(f"- Launcher runtime (HH:MM:SS): {_format_hms(launcher_runtime_sec)}")
    print(f"- Execution device: {manifest.get('execution_device', 'N/A')}")
    print(f"- Acceleration backend: {manifest.get('acceleration_backend', 'N/A')}")
    print()
    print(f"Validation result: {validation_status}")


def main() -> None:
    _maybe_reexec_in_project_venv()
    ensure_structure()
    host_profile = _detect_host_profile()
    cpu_default = max(1, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(description="Run full Soft Computing project pipeline.")
    parser.add_argument("--fresh", action="store_true", help="Start benchmark from scratch.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap.")
    parser.add_argument("--checkpoint-every", type=int, default=120)
    parser.add_argument("--checkpoint-percent", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=cpu_default, help="Parallel model workers.")
    parser.add_argument("--selection-jobs", type=int, default=cpu_default, help="Parallel SFS workers.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial validation mode.")
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default=None,
        help="Execution mode for benchmark.",
    )
    # Backward-compatible alias used by older docs/scripts.
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--platform-profile",
        choices=["auto", "linux", "mac", "windows"],
        default="auto",
        help="Target OS profile label for reproducibility notes.",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Fail when GPU mode is requested but unavailable.",
    )
    parser.add_argument(
        "--show-convergence-warnings",
        action="store_true",
        help="Show sklearn convergence warnings (hidden by default).",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable prompts and require all choices via flags/defaults.",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode is None and args.device in {"cpu", "gpu"}:
        mode = args.device

    platform_profile = args.platform_profile
    if platform_profile == "auto":
        platform_profile = host_profile

    if not args.non_interactive:
        if mode is None:
            mode = _prompt_choice("Choose execution mode", ["cpu", "gpu"], default="cpu")
        platform_profile = _prompt_choice(
            "Choose platform profile (for run metadata and docs)",
            ["linux", "windows", "mac"],
            default=platform_profile,
        )

    if mode is None:
        mode = "cpu"

    # Keep matplotlib in non-GUI backend to avoid tkinter/thread cleanup errors.
    run_env = os.environ.copy()
    run_env.setdefault("MPLBACKEND", "Agg")
    run_env.setdefault("PYTHONUNBUFFERED", "1")

    ctx = _create_run_context(mode=mode, platform_profile=platform_profile)
    py = sys.executable

    step_times: Dict[str, float] = {}
    step_times["fetch_data"] = _run_step("Fetch dataset", [py, "src/cli/fetch_data.py"], env=run_env)

    check_cmd = [py, "src/cli/check_env.py", "--device", mode]
    if args.strict_device:
        check_cmd.append("--strict-device")
    step_times["check_env"] = _run_step("Check environment", check_cmd, env=run_env)

    run_cmd = [
        py,
        "src/cli/run_experiments.py",
        "--checkpoint-every",
        str(max(1, args.checkpoint_every)),
        "--checkpoint-percent",
        str(max(1, min(100, args.checkpoint_percent))),
        "--jobs",
        str(max(1, args.jobs)),
        "--selection-jobs",
        str(max(1, args.selection_jobs)),
        "--device",
        mode,
        "--platform-profile",
        platform_profile,
        "--run-label",
        ctx.run_id,
    ]
    if args.strict_device:
        run_cmd.append("--strict-device")
    if args.fresh:
        run_cmd.append("--fresh")
    if args.max_rows is not None:
        run_cmd.extend(["--max-rows", str(args.max_rows)])
    if args.show_convergence_warnings:
        run_cmd.append("--show-convergence-warnings")
    step_times["run_experiments"] = _run_step("Run benchmark", run_cmd, env=run_env)

    validate_cmd = [py, "src/cli/validate_cartesian_outputs.py"]
    if args.allow_partial:
        validate_cmd.append("--allow-partial")
    step_times["validate"] = _run_step("Validate outputs", validate_cmd, env=run_env)

    step_times["paper_drafts"] = _run_step(
        "Generate paper drafts",
        [py, "src/cli/generate_paper_drafts.py"],
        env=run_env,
    )

    manifest, validation_status = _archive_run_outputs(ctx=ctx, step_times=step_times)
    _update_run_history(ctx=ctx, step_times=step_times, validation_status=validation_status, manifest=manifest)
    _print_validation_block(
        manifest=manifest,
        validation_status=validation_status,
        allow_partial=args.allow_partial,
        launcher_runtime_sec=sum(step_times.values()),
    )

    print("\nAll steps completed.")
    print(f"- Run id: {ctx.run_id}")
    print(f"- Run archive: {ctx.run_dir}")
    print(f"- Run history: {RESULTS_HISTORY_INDEX_MD}")


if __name__ == "__main__":
    main()
