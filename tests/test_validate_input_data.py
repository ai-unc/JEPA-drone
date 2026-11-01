"""Tests for skyjepa.data.validate_input_data on synthetic logs with known faults.

    uv run pytest tests/test_validate_input_data.py
"""

import json

import numpy as np
import pytest

from skyjepa.config import Config
from skyjepa.data import trajectory_loader

try:
    from skyjepa.data.dataset import build_dataset
except ImportError:  # fix-training branch moves it into the Data class
    from skyjepa.data.dataset import Data

    build_dataset = Data.build_dataset
from skyjepa.data.validate_input_data import (
    DEFAULT_CONFIG,
    HEADER,
    Limits,
    check_file,
    min_rows,
    validate_folder,
)

DT = 0.01


# ----------------------------------------------------------------- synthetic logs


def smooth_log(n: int = 1000, vz_amplitude: float = 5.0) -> np.ndarray:
    """A level drone that moves up and down smoothly. Kinematics are consistent."""

    t = np.arange(n) * DT
    vz = vz_amplitude * np.sin(2 * np.pi * t / 10.0)
    z = -100.0 + np.concatenate([[0.0], np.cumsum((vz[1:] + vz[:-1]) / 2 * DT)])
    data = np.zeros((n, len(HEADER)))
    data[:, 0] = t
    data[:, 1:5] = 2.6
    data[:, 7] = z
    data[:, 10] = vz
    data[:, 11:20] = np.eye(3).ravel()
    return data


def flip_log() -> np.ndarray:
    """A full, smooth roll (360 degrees in 2 s). Rotation and body rate agree."""

    data = smooth_log()
    t = np.arange(len(data)) * DT
    s = np.clip((t - 4.0) / 2.0, 0.0, 1.0)
    angle = 2 * np.pi * (3 * s**2 - 2 * s**3)
    rate = 2 * np.pi * (6 * s - 6 * s**2) / 2.0
    c, n = np.cos(angle), np.sin(angle)
    rotation = np.zeros((len(data), 3, 3))
    rotation[:, 0, 0] = 1
    rotation[:, 1, 1], rotation[:, 1, 2], rotation[:, 2, 1], rotation[:, 2, 2] = c, -n, n, c
    data[:, 11:20] = rotation.reshape(-1, 9)
    data[:, 20] = rate
    return data


def write_log(path, data, config=None):
    lines = []
    if config is not None:
        lines.append("# CONFIG: " + json.dumps(config))
    lines.append(",".join(HEADER))
    lines += [",".join(repr(float(v)) for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n")


def check(tmp_path, data, config=None):
    path = tmp_path / "log.csv"
    write_log(path, data, config)
    return check_file(path, tmp_path, Limits(), Config())


# ----------------------------------------------------------------- checks


def test_clean_log_passes(tmp_path):
    result = check(tmp_path, smooth_log())
    assert result.status == "passed" and result.rule == "" and result.row is None


def test_rate_blow_up_is_rejected(tmp_path):
    data = smooth_log()
    data[600:, 20] = 3e8  # logged rate explodes, rotation does not follow
    result = check(tmp_path, data)
    assert (result.status, result.rule, result.row) == ("rejected", "rotation blow-up", 600)


def test_impact_is_rejected(tmp_path):
    data = smooth_log()
    data[400:, 10] += 50.0  # velocity jump of 50 m/s in one step
    data[400:, 7] += np.arange(600) * DT * 50.0 + 0.25  # keep position consistent
    result = check(tmp_path, data)
    assert (result.rule, result.row) == ("impact", 400)


def test_nan_is_rejected(tmp_path):
    data = smooth_log()
    data[300, 8] = np.nan
    result = check(tmp_path, data)
    assert (result.rule, result.row) == ("non-finite value", 300)


def test_fast_flight_is_rejected_for_speed(tmp_path):
    result = check(tmp_path, smooth_log(vz_amplitude=30.0))  # |v| > 25 m/s near each peak
    assert result.rule == "speed" and "> 25" in result.detail


def test_inverted_flight_is_rejected_for_tilt(tmp_path):
    result = check(tmp_path, flip_log())  # valid physics, but upside down
    assert result.rule == "tilt" and result.max_tilt_deg > 179


def test_rate_limit_is_never_above_the_sampling_limit(tmp_path):
    path = tmp_path / "log.csv"
    write_log(path, flip_log())
    loose = Limits(max_rate=1000.0, max_tilt_deg=180.0)
    assert check_file(path, tmp_path, loose, Config()).status == "passed"  # peak rate ~4.7 rad/s
    data = smooth_log()
    data[:, 22] = 70.0  # constant yaw rate above pi / 0.05 s
    angle = 70.0 * np.arange(len(data)) * DT
    rotation = np.zeros((len(data), 3, 3))
    rotation[:, 2, 2] = 1
    rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 1, 0], rotation[:, 1, 1] = (
        np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle))
    data[:, 11:20] = rotation.reshape(-1, 9)
    write_log(path, data)
    assert check_file(path, tmp_path, loose, Config()).rule == "rate"


@pytest.mark.parametrize(
    "config, text",
    [({**DEFAULT_CONFIG, "mass": 5.0}, "outside"), ({"mass": 1.22}, "keys")],
)
def test_bad_config_line_is_rejected(tmp_path, config, text):
    result = check(tmp_path, smooth_log(), config)
    assert result.rule == "format" and text in result.detail


def test_short_file_is_rejected(tmp_path):
    result = check(tmp_path, smooth_log(min_rows(Config()) - 1))
    assert result.rule == "format" and "fewer than one window" in result.detail


# ----------------------------------------------------------------- folders and output


def make_inputs(tmp_path):
    source = tmp_path / "in" / "set"
    source.mkdir(parents=True)
    write_log(source / "clean.csv", smooth_log())
    write_log(source / "fast.csv", smooth_log(vz_amplitude=30.0))  # rejected: speed
    write_log(source / "flip.csv", flip_log())  # rejected: tilt
    return tmp_path / "in", tmp_path / "out", source


def test_only_passed_files_are_copied_unchanged(tmp_path):
    root, out, source = make_inputs(tmp_path)
    results = {r.file: r for r in validate_folder(root, out)}
    assert results["clean.csv"].status == "passed"
    assert results["fast.csv"].status == results["flip.csv"].status == "rejected"
    assert sorted(p.name for p in (out / "set").iterdir()) == ["clean.csv"]
    assert (out / "set" / "clean.csv").read_bytes() == (source / "clean.csv").read_bytes()


def test_shortest_passed_file_loads_into_one_window(tmp_path):
    config = Config()
    source = tmp_path / "in"
    source.mkdir()
    write_log(source / "one.csv", smooth_log(min_rows(config)), DEFAULT_CONFIG)
    validate_folder(source, tmp_path / "out")
    trajectories = trajectory_loader.load_trajectories(tmp_path / "out", stride=trajectory_loader.stride_for(config.dt))
    windows = build_dataset(trajectories, history_len=config.history_len, horizon=config.horizon)
    assert windows["state_history"].shape[0] == 1


def test_dry_run_writes_nothing(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    results = validate_folder(root, None)
    assert len(results) == 3 and not out.exists()


# ----------------------------------------------------------------- incremental runs


def test_second_run_skips_every_file(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    first = validate_folder(root, out)
    before = (out / "set" / "clean.csv").stat().st_mtime_ns
    messages = []
    second = validate_folder(root, out, log=messages.append)
    assert all(r.action == "skipped" for r in second)
    assert len(messages) == 3 and all("skipped" in m for m in messages)
    assert [r.status for r in second] == [r.status for r in first]  # earlier results reused
    assert (out / "set" / "clean.csv").stat().st_mtime_ns == before  # not copied again


def test_changed_file_is_checked_again(tmp_path):
    root, out, source = make_inputs(tmp_path)
    validate_folder(root, out)
    write_log(source / "clean.csv", smooth_log(vz_amplitude=30.0))  # now it fails
    messages = []
    results = {r.file: r for r in validate_folder(root, out, log=messages.append)}
    assert results["clean.csv"].action == "checked" and results["clean.csv"].status == "rejected"
    assert not (out / "set" / "clean.csv").exists()  # the old passed copy is removed
    assert any("changed" in m for m in messages)
    assert results["fast.csv"].action == "skipped"


def test_force_checks_every_file_again(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    validate_folder(root, out)
    results = validate_folder(root, out, force=True)
    assert all(r.action == "checked" for r in results)
    assert (out / "set" / "clean.csv").exists()


def test_other_settings_need_force(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    validate_folder(root, out)
    with pytest.raises(ValueError, match="other settings"):
        validate_folder(root, out, Limits(max_speed=40.0))
    results = {r.file: r for r in validate_folder(root, out, Limits(max_speed=40.0), force=True)}
    assert results["fast.csv"].status == "passed"  # 30 m/s is inside the new limit


def test_output_without_manifest_is_skipped_by_name(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    (out / "set").mkdir(parents=True)
    (out / "set" / "clean.csv").write_text("from an older run\n")
    messages = []
    results = {r.file: r for r in validate_folder(root, out, log=messages.append)}
    assert results["clean.csv"].action == "skipped" and results["clean.csv"].status == "unknown"
    assert (out / "set" / "clean.csv").read_text() == "from an older run\n"
    assert results["fast.csv"].action == "checked"
    assert any("set/clean.csv skipped (an output with this name exists)" in m for m in messages)


def test_missing_output_is_copied_again(tmp_path):
    root, out, _ = make_inputs(tmp_path)
    validate_folder(root, out)
    (out / "set" / "clean.csv").unlink()
    results = {r.file: r for r in validate_folder(root, out)}
    assert results["clean.csv"].action == "checked" and (out / "set" / "clean.csv").exists()
