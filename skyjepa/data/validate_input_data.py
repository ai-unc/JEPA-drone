"""Validate simulator CSV logs: pass a file only if every row is safe to train on.

    uv run python -m skyjepa.data.validate_input_data drone/data_sequences drone/validated_sequences
    uv run python -m skyjepa.data.validate_input_data drone/data_sequences drone/validated_sequences --force
    uv run python -m skyjepa.data.validate_input_data drone/data_sequences --dry-run --report report.csv

A file passes only if it passes every check below. A passed file is copied to
the output folder unchanged. A file that fails any check is rejected whole and
is not copied. A rejection is a signal that the data generator has a problem:
the report gives the first rule that failed, the row and the value.

1. Format checks (whole file)
   Header, optional "# CONFIG:" line (valid JSON, every value in the
   SimulationConfigs ranges), parsable rows, enough rows for one window.
2. Simulation checks (each row and each 10 ms step)
   Non-finite value, wrong time step, invalid rotation matrix, impact,
   position blow-up, rotation blow-up.
3. Flight envelope checks (each row)
   Body rate, speed, tilt. Tilt above 90 degrees is upside down, so the tilt
   limit rejects every file in which the drone is inverted.

Incremental runs: the output folder keeps validation_manifest.json, with the
settings of the run and, for each input file, its SHA-256 hash and its result.
A later run into the same folder skips each input file that is unchanged
("skipped"). A changed file is checked again, and its old output is replaced
or removed. An input with an output of the same name but no manifest entry is
also skipped. --force checks every file again. A run with different settings
stops, unless --force is given, so that one folder never mixes data from
different limits.

Types: this module reads CSV text and copies files. trajectory_loader reads the
passed CSVs into np.float32 arrays, and train.WindowDataset converts the
windows to torch.float32 tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from skyjepa.config import Config
from skyjepa.data.trajectory_loader import ACTION_COLUMNS, LOG_HZ, STATE_COLUMNS, stride_for

# =============================================================================
# Constants: the CSV layout, the physics and the limits
# =============================================================================

# The 23 columns that FlightController.write_data_log writes, in order.
HEADER = ("time", *ACTION_COLUMNS, *STATE_COLUMNS)
CONFIG_PREFIX = "# CONFIG:"
LOG_DT = 1.0 / LOG_HZ  # 0.01 s between rows
GRAVITY = np.array([0.0, 0.0, 9.81])  # NED world frame: +z points down
LOGGED_MAX_THRUST = 4.18  # N per motor at the default thrust_coef (Action.ASSUMED_MAX_THRUST)

# Column positions in HEADER, so that the checks can slice the data array.
POSITION = slice(5, 8)  # x, y, z (m)
VELOCITY = slice(8, 11)  # vx, vy, vz (m/s)
ROTATION = slice(11, 20)  # body-to-world rotation matrix, row-major
RATE = slice(20, 23)  # wx, wy, wz (rad/s), body frame
R33 = 19  # R[2, 2]: world-z component of the body z axis, equal to cos(tilt)

# Default physics and the SimulationConfigs.randomize() ranges.
# Keep in sync with drone/simulations/simulation_config.py. This module does not
# import that file, so that training code does not depend on the simulator
# package (projectairsim, json5).
DEFAULT_CONFIG = {
    "mass": 1.22,
    "thrust_coef": 0.109919,
    "torque_coef": 0.040164,
    "arm_length": 0.3578,
    "motor_time_constant": 0.005,
    "drag_coef": 0.325,
}
CONFIG_RANGES = {
    "mass": (0.5 * 1.22, 1.5 * 1.22),
    "thrust_coef": (0.1, 0.28),
    "torque_coef": (0.5 * 0.040164, 1.5 * 0.040164),
    "arm_length": (0.5 * 0.3578, 1.5 * 0.3578),
    "motor_time_constant": (0.5 * 0.005, 1.5 * 0.005),
    "drag_coef": (0.0, 0.325),
}


@dataclass(frozen=True)
class Limits:
    """Every threshold the validator uses. Each field is also a CLI flag.

    The comment after each limit gives the value in clean data (432 logs,
    November 2026), so that the margin is clear.
    """

    # Simulation checks: is the simulation still valid?
    max_orthonormal_error: float = 1e-3  # ||R R^T - I||; clean data ~1e-15
    max_force_ratio: float = 4.0  # |a - g| / max thrust acceleration; clean p99.99 = 3.2
    max_position_residual: float = 0.5  # m/s; clean p99.9 = 0.03
    max_rotation_residual_deg: float = 1.0  # per 10 ms step; clean p99 < 0.2
    # Flight envelope checks: is this realistic flight?
    max_rate: float = 10.0  # rad/s; never used above 0.95 * pi / model dt
    max_speed: float = 25.0  # m/s
    max_tilt_deg: float = 60.0  # degrees between body z and world down


@dataclass
class FileResult:
    """One row of the report: what happened to one input file."""

    dataset: str  # subfolder of the input folder
    file: str
    action: str  # "checked" in this run, or "skipped" (result from an earlier run)
    status: str  # "passed", "rejected" or "unknown" (skipped by name, no manifest entry)
    rule: str  # the first check that failed; empty if passed
    row: int | None  # data row (0 = first row after the header) where it failed
    detail: str  # the value that failed, e.g. "|w| = 12.3 rad/s > 10"
    config: str  # "line", "default (no line)" or "invalid"
    rows: int
    max_rate: float  # rad/s, whole file (diagnostics for the generator)
    max_speed: float  # m/s, whole file
    max_tilt_deg: float  # degrees, whole file; above 90 means upside down


class FormatError(Exception):
    """The file fails a format check."""


# =============================================================================
# 1. Format checks: read the file
# =============================================================================


def read_log(path: Path) -> tuple[str | None, np.ndarray]:
    """Read one log. Returns (config line or None, data array of shape (n, 23))."""

    lines = path.read_text().splitlines()
    config_line = None
    if lines and lines[0].startswith("#"):
        config_line, lines = lines[0], lines[1:]
    if not lines:
        raise FormatError("no header")
    header = tuple(name.strip() for name in lines[0].split(","))
    if header != HEADER:
        raise FormatError(f"header is not the 23-column log header: {lines[0][:80]}")
    values = []
    for number, line in enumerate((line for line in lines[1:] if line.strip()), start=0):
        cells = line.split(",")
        if len(cells) != len(HEADER):
            raise FormatError(f"data row {number} has {len(cells)} columns")
        try:
            values.append([float(cell) for cell in cells])  # "nan" parses; a simulation check catches it
        except ValueError:
            raise FormatError(f"data row {number} has a value that is not a number") from None
    return config_line, np.array(values, dtype=np.float64).reshape(-1, len(HEADER))


def parse_config(config_line: str | None) -> tuple[dict[str, float], str]:
    """Return the physics of the file and how it was found.

    A file with no config line used the default physics.
    """

    if config_line is None:
        return dict(DEFAULT_CONFIG), "default (no line)"
    if not config_line.startswith(CONFIG_PREFIX):
        raise FormatError("first line starts with '#' but is not a '# CONFIG:' line")
    try:
        values = json.loads(config_line[len(CONFIG_PREFIX):])
    except json.JSONDecodeError as error:
        raise FormatError(f"config line is not valid JSON ({error.msg})") from None
    if set(values) != set(DEFAULT_CONFIG):
        raise FormatError(f"config keys {sorted(values)} are not {sorted(DEFAULT_CONFIG)}")
    for key, (low, high) in CONFIG_RANGES.items():
        value = float(values[key])
        # Small tolerance: the simulator rounds config values to 7 digits.
        if not math.isfinite(value) or not (low - 1e-6 <= value <= high + 1e-6):
            raise FormatError(f"config {key}={value} is outside [{low:.6g}, {high:.6g}]")
    return {key: float(values[key]) for key in DEFAULT_CONFIG}, "line"


def min_rows(config: Config) -> int:
    """Log rows needed for one (history, horizon) training window.

    One window has history_len + horizon states, so (H + T - 1) model steps of
    `stride` log rows, plus the first row. With H = T = 10 at 20 Hz: 96 rows.
    """

    return (config.history_len + config.horizon - 1) * stride_for(config.dt) + 1


# =============================================================================
# 2 and 3. Row and step checks
# =============================================================================


def _rotation_step(rates: np.ndarray) -> np.ndarray:
    """exp([w]x * dt) for each row of body rates (n, 3), by the Rodrigues formula.

    This is the rotation that the drone makes in one 10 ms step at rate w.
    """

    angle = np.linalg.norm(rates, axis=1) * LOG_DT
    axis = rates / np.where(angle == 0, 1.0, angle / LOG_DT)[:, None]  # unit axis; 0 if no rotation
    x, y, z = axis.T
    k = np.zeros((len(rates), 3, 3))  # skew-symmetric matrix of the axis
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 0] = -z, y, z
    k[:, 1, 2], k[:, 2, 0], k[:, 2, 1] = -x, -y, x
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def max_specific_force(config: dict[str, float]) -> float:
    """Largest acceleration (m/s^2) the four motors can give this drone.

    Uses the logged thrust scale. The real thrust is about 1.13 times the
    logged command, and the 4x limit has room for that and for drag.
    """

    max_thrust = LOGGED_MAX_THRUST * config["thrust_coef"] / DEFAULT_CONFIG["thrust_coef"]
    return 4 * max_thrust / config["mass"]


def effective_max_rate(limits: Limits, model_dt: float) -> float:
    """The rate limit, never above 0.95 * pi / model dt.

    Above pi / model dt the drone turns more than 180 degrees between two model
    samples, so the data cannot show the direction of the rotation.
    """

    return min(limits.max_rate, 0.95 * math.pi / model_dt)


def tilt_deg(data: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and world down, in degrees. Above 90 is upside down."""

    return np.degrees(np.arccos(np.clip(data[:, R33], -1.0, 1.0)))


def first_failure(
    data: np.ndarray, physics: dict[str, float], limits: Limits, model_dt: float
) -> tuple[int, str, str] | None:
    """Return (row, rule, detail) of the earliest failed check, or None if all pass.

    Each check gives one value for each row (or for each step from row i to
    row i + 1; a step failure is reported at row i + 1, where the bad value is
    seen). If two checks fail at the same row, the first one in the list below
    is reported, so that the most specific cause is shown (at an impact, the
    rotation check also fails).
    """

    rotation = data[:, ROTATION].reshape(-1, 3, 3)
    velocity, rate = data[:, VELOCITY], data[:, RATE]
    with np.errstate(invalid="ignore", over="ignore"):  # NaN rows fail the first check
        # Values for each step, padded with 0 at row 0 so that index = row.
        def step(values: np.ndarray) -> np.ndarray:
            return np.concatenate([[0.0], values])

        predicted = rotation[:-1] @ _rotation_step((rate[1:] + rate[:-1]) / 2)
        cosine = (np.einsum("nij,nij->n", rotation[1:], predicted) - 1) / 2
        checks = [
            # (rule, value for each row, limit, unit). The row fails if value > limit.
            ("non-finite value", (~np.isfinite(data).all(axis=1)).astype(float), 0.0, ""),
            ("time step", step(np.abs(np.diff(data[:, 0]) - LOG_DT)), 1e-6, "s"),
            ("rotation not orthonormal",
             np.linalg.norm(rotation @ rotation.transpose(0, 2, 1) - np.eye(3), axis=(1, 2)),
             limits.max_orthonormal_error, ""),
            # Impact: the acceleration minus gravity must be possible with the motors.
            ("impact",
             step(np.linalg.norm(np.diff(velocity, axis=0) / LOG_DT - GRAVITY, axis=1) / max_specific_force(physics)),
             limits.max_force_ratio, "x max thrust"),
            # Position blow-up: the position change must equal the mean velocity times 10 ms.
            ("position blow-up",
             step(np.linalg.norm(np.diff(data[:, POSITION], axis=0) / LOG_DT - (velocity[1:] + velocity[:-1]) / 2, axis=1)),
             limits.max_position_residual, "m/s"),
            # Rotation blow-up: R[i+1] must equal R[i] turned by the mean body rate for 10 ms.
            ("rotation blow-up", step(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
             limits.max_rotation_residual_deg, "deg"),
            ("rate", np.linalg.norm(rate, axis=1), effective_max_rate(limits, model_dt), "rad/s"),
            ("speed", np.linalg.norm(velocity, axis=1), limits.max_speed, "m/s"),
            ("tilt", tilt_deg(data), limits.max_tilt_deg, "deg"),
        ]

    earliest = None
    for rule, values, limit, unit in checks:
        bad = ~(values <= limit)  # NaN counts as a failure
        if bad.any():
            row = int(np.argmax(bad))  # first failed row
            if earliest is None or row < earliest[0]:
                if rule == "non-finite value":
                    detail = "NaN or Inf in this row"
                else:
                    detail = f"{values[row]:.4g} {unit} > {limit:.4g}".replace("  ", " ")
                earliest = (row, rule, detail)
    return earliest


# =============================================================================
# One file
# =============================================================================


def check_file(path: Path, root: Path, limits: Limits, config: Config) -> FileResult:
    """Run every check on one log and return its result. Writes nothing."""

    relative = path.relative_to(root)
    result = FileResult(
        dataset=str(relative.parent), file=relative.name, action="checked", status="rejected",
        rule="", row=None, detail="", config="invalid", rows=0,
        max_rate=float("nan"), max_speed=float("nan"), max_tilt_deg=float("nan"),
    )

    # 1. Format checks.
    try:
        config_line, data = read_log(path)
        result.rows = len(data)
        physics, result.config = parse_config(config_line)
        if len(data) < min_rows(config):
            raise FormatError(f"{len(data)} rows, fewer than one window ({min_rows(config)})")
    except FormatError as error:
        result.rule, result.detail = "format", str(error)
        return result

    # Diagnostics for the report: the largest values in the file.
    with np.errstate(invalid="ignore"):
        result.max_rate = round(float(np.nanmax(np.linalg.norm(data[:, RATE], axis=1))), 3)
        result.max_speed = round(float(np.nanmax(np.linalg.norm(data[:, VELOCITY], axis=1))), 3)
        result.max_tilt_deg = round(float(np.nanmax(tilt_deg(data))), 2)

    # 2 and 3. Simulation and envelope checks.
    failure = first_failure(data, physics, limits, config.dt)
    if failure is None:
        result.status = "passed"
    else:
        result.row, result.rule, result.detail = failure
    return result


# =============================================================================
# Incremental runs: the manifest in the output folder
# =============================================================================

MANIFEST = "validation_manifest.json"


def file_digest(path: Path) -> str:
    """SHA-256 of the file bytes. A changed file gets a different digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_settings(limits: Limits, config: Config) -> dict:
    """Every setting that changes the result. Two runs agree only if these agree."""

    return {
        "limits": asdict(limits),
        "effective_max_rate": effective_max_rate(limits, config.dt),
        "model": {"dt": config.dt, "history_len": config.history_len, "horizon": config.horizon},
        "min_rows": min_rows(config),
    }


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {"settings": None, "files": {}}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    """Write to a temporary file, then rename, so that a stopped run never leaves half a manifest."""

    temporary = out_dir / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=1) + "\n")
    temporary.replace(out_dir / MANIFEST)


# =============================================================================
# A folder
# =============================================================================


def validate_folder(
    root: Path,
    out_dir: Path | None,
    limits: Limits = Limits(),
    config: Config = Config(),
    force: bool = False,
    log=print,
) -> list[FileResult]:
    """Check every *.csv under root (recursively) and copy the passed files to out_dir.

    The output keeps the input subfolders: root/a/b.csv is copied to out_dir/a/b.csv.
    With out_dir=None (dry run), nothing is written and nothing is skipped.
    force=True checks every file again. log receives one message for each
    skipped or changed file.
    """

    own_output = out_dir.resolve() if out_dir is not None else None  # never read our own output
    paths = sorted(
        (path for path in root.rglob("*.csv") if own_output is None or own_output not in path.resolve().parents),
        key=lambda path: (str(path.parent), len(path.stem), path.stem),  # rnd_sky2 before rnd_sky10
    )
    if not paths:
        raise FileNotFoundError(f"no CSV logs under {root}")
    if out_dir is None:
        return [check_file(path, root, limits, config) for path in paths]

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = run_settings(limits, config)
    manifest = load_manifest(out_dir)
    if manifest["settings"] not in (None, settings):
        if not force:
            raise ValueError(
                f"{out_dir} was made with other settings; use --force to check everything "
                "again with the new settings, or give a new folder"
            )
        manifest["files"] = {}  # old results do not apply to the new settings
    manifest["settings"] = settings

    results = []
    for path in paths:
        name = str(path.relative_to(root))  # manifest key, e.g. "rnd_sky_training/rnd_sky1.csv"
        output = out_dir / name
        digest = file_digest(path)
        entry = manifest["files"].get(name)

        if not force:
            # Same content as in the manifest, and the output is there if it passed: skip.
            if entry is not None and entry["sha256"] == digest:
                if entry["result"]["status"] != "passed" or output.exists():
                    log(f"{name} skipped (already checked)")
                    results.append(FileResult(**{**entry["result"], "action": "skipped"}))
                    continue
                log(f"{name} passed before, but its output is missing; checked again")
            # Not in the manifest, but an output with its name exists: skip by name.
            elif entry is None and output.exists():
                log(f"{name} skipped (an output with this name exists)")
                results.append(FileResult(
                    dataset=str(path.parent.relative_to(root)), file=path.name, action="skipped",
                    status="unknown", rule="", row=None, detail="output exists from a run with no manifest",
                    config="", rows=0, max_rate=float("nan"), max_speed=float("nan"), max_tilt_deg=float("nan"),
                ))
                continue
            elif entry is not None:
                log(f"{name} changed since the last run; checked again")

        result = check_file(path, root, limits, config)
        if result.status == "passed":
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)  # the exact input bytes
        elif output.exists():
            output.unlink()  # it passed before, but not now: remove the old copy
        results.append(result)
        manifest["files"][name] = {"sha256": digest, "result": asdict(result)}
        save_manifest(out_dir, manifest)  # after each file, so that a stopped run loses nothing
    return results


# =============================================================================
# Report and summary
# =============================================================================


def write_report(results: list[FileResult], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(FileResult)])
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def summary(results: list[FileResult]) -> str:
    """Files passed and rejected for each dataset, then the rejections by rule."""

    groups: dict[str, list[FileResult]] = defaultdict(list)
    for result in results:
        groups[result.dataset].append(result)
    lines = [f"{'dataset':34s} {'files':>5s} {'skipped':>7s} {'passed':>6s} {'rejected':>8s}"]
    for name, group in sorted(groups.items()):
        lines.append(
            f"{name[:34]:34s} {len(group):5d} {sum(r.action == 'skipped' for r in group):7d} "
            f"{sum(r.status == 'passed' for r in group):6d} {sum(r.status == 'rejected' for r in group):8d}"
        )
    rules = Counter(r.rule for r in results if r.status == "rejected")
    if rules:
        lines.append("rejected by rule: " + ", ".join(f"{rule} {count}" for rule, count in rules.most_common()))
    return "\n".join(lines)


# =============================================================================
# Command line
# =============================================================================


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulator CSV logs and copy the files that pass.")
    parser.add_argument("input", type=Path, help="folder of CSV logs (searched recursively)")
    parser.add_argument("output", type=Path, nargs="?", help="folder for the passed files (new, or from an earlier run)")
    parser.add_argument("--dry-run", action="store_true", help="check and report only; copy nothing")
    parser.add_argument("--force", action="store_true", help="check every file again, also files checked before")
    parser.add_argument("--report", type=Path, help="report CSV path (default: OUTPUT/validation_report.csv)")
    for field in fields(Limits):  # one flag for each limit, e.g. --max-rate 20
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=float, default=field.default)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    limits = Limits(**{field.name: getattr(args, field.name) for field in fields(Limits)})
    config = Config()
    if not args.dry_run and args.output is None:
        raise SystemExit("give an OUTPUT folder, or use --dry-run")
    out_dir = None if args.dry_run else args.output

    try:
        results = validate_folder(args.input, out_dir, limits, config, force=args.force)
    except ValueError as error:  # settings differ from the earlier run
        raise SystemExit(str(error)) from None

    # The report always lists every input file; skipped files keep their earlier result.
    report = args.report or (out_dir / "validation_report.csv" if out_dir else None)
    if report is not None:
        write_report(results, report)

    print(summary(results))
    print(f"rate limit used: {effective_max_rate(limits, config.dt):.3g} rad/s; min rows: {min_rows(config)}")
    if report is not None:
        print(f"report: {report}")


if __name__ == "__main__":
    main()
