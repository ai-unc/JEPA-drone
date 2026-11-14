"""Random primitive geometry for depth collection on a flat Project AirSim map.

Coordinates and sizes are in meters in NED (negative z is up); rotations are
roll/pitch/yaw in degrees. Sizes assume centered, unit-size primitive assets.
Simulation code is responsible for spawning these records into the world.

Usage::

    course = ObstacleCourse()
    course.randomize()
    # After the flight logger has finished writing the CSV:
    course.write_to_csv("recording.csv")
    replay_course = ObstacleCourse()
    replay_course.read_csv("recording.csv")

The extra comment header preserves the first-line simulation config and is
ignored by the existing pandas reader (comment="#"). For logging directly,
append course.csv_header() to the simulation's existing csv_header() result.
"""

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


class ObstacleCourse:
    class Shape(Enum):
        RECTANGLE = "Cube"
        SPHERE = "Sphere"
        DISK = "Cylinder"

    @dataclass
    class Obstacle:
        shape: "ObstacleCourse.Shape"
        size: np.ndarray
        position: np.ndarray
        rotation: np.ndarray

        def __post_init__(self):
            self.shape = ObstacleCourse.Shape(self.shape)
            for name in ("size", "position", "rotation"):
                vector = np.array(getattr(self, name), dtype=float, copy=True)
                if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                    raise ValueError(f"{name} must contain three finite numbers")
                setattr(self, name, vector)
            if np.any(self.size <= 0):
                raise ValueError("Obstacle sizes must be positive")

        def to_dict(self):
            return dict(shape=self.shape.value, size=self.size.tolist(),
                        position=self.position.tolist(), rotation=self.rotation.tolist())

        def to_string(self) -> str:
            return json.dumps(self.to_dict(), allow_nan=False)

        def relative_height(self) -> float:
            """Full vertical extent after Rz(yaw) @ Ry(pitch) @ Rx(roll).

            Project local axes onto world z. Yaw does not affect height.
            Use each primitive's actual extent rather than its bounding box.
            """
            roll, pitch = np.deg2rad(self.rotation[:2])
            z_axes = np.array([-np.sin(pitch),
                               np.cos(pitch) * np.sin(roll),
                               np.cos(pitch) * np.cos(roll)])
            projected = self.size * z_axes
            if self.shape == ObstacleCourse.Shape.RECTANGLE:
                return float(np.abs(projected).sum())
            if self.shape == ObstacleCourse.Shape.SPHERE:
                return float(np.linalg.norm(projected))
            # Cylinder's elliptical radial cross section plus axial extent.
            return float(np.linalg.norm(projected[:2]) + abs(projected[2]))

    HEADER = "OBSTACLE_COURSE"
    CSV_PREFIX = "# " + HEADER + ": "

    def __init__(self):
        self.shapes: list[ObstacleCourse.Obstacle] = []
        self.min_angles = -90
        self.max_angles = 90
        self.min_x, self.max_x = -150, 150
        self.min_y, self.max_y = -150, 150
        self.ground_level = 0
        self.min_size, self.max_size = 10, 40
        self.rectangles = 30
        self.spheres = 10
        self.disks = 20

    def randomize(self) -> None:
        """Replace geometry with random sizes, positions, and roll/pitch/yaw.

        Centers range from ground level to half the rotated vertical extent
        above ground, allowing partially buried obstacles without floating.

        Placement is independent: overlaps are permitted. A local random
        generator leaves the flight's random action sequence untouched.
        Spheres use equal diameters; cylinders use equal x/y diameters.
        """
        ranges = ((self.min_x, self.max_x), (self.min_y, self.max_y),
                  (self.min_size, self.max_size), (self.min_angles, self.max_angles))
        if any(not np.all(np.isfinite(pair)) or pair[0] > pair[1] for pair in ranges):
            raise ValueError("Randomization bounds must be finite and ordered")
        if self.min_size <= 0 or not np.isfinite(self.ground_level):
            raise ValueError("Sizes must be positive and ground_level finite")
        counts = (self.rectangles, self.spheres, self.disks)
        if any(not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
            raise ValueError("Shape counts must be nonnegative integers")
        rng = np.random.default_rng()
        shapes = []
        for kind, count in zip(self.Shape, counts):
            for _ in range(count):
                size = rng.uniform(self.min_size, self.max_size, 3)
                if kind == self.Shape.SPHERE:
                    size[:] = size[0]
                elif kind == self.Shape.DISK:
                    size[1] = size[0]
                rotation = rng.uniform(self.min_angles, self.max_angles, 3)
                obstacle = self.Obstacle(kind, size, [0, 0, self.ground_level], rotation)
                obstacle.position[:] = [
                    rng.uniform(self.min_x, self.max_x),
                    rng.uniform(self.min_y, self.max_y),
                    rng.uniform(self.ground_level - obstacle.relative_height() / 2,
                                self.ground_level),
                ]
                shapes.append(obstacle)
        self.shapes = shapes

    def clear(self) -> None:
        """Clear stored geometry; world objects must be removed by simulation code."""
        self.shapes.clear()

    def to_string(self) -> str:
        """Serialize exact geometry for replay."""
        return json.dumps({"version": 1, "shapes": [s.to_dict() for s in self.shapes]},
                          allow_nan=False, separators=(",", ":"))

    def csv_header(self) -> str:
        return self.CSV_PREFIX + self.to_string() + "\n"

    def write_to_csv(self, PATH: str) -> None:
        """Insert/replace metadata after leading comments, before CSV columns.

        Call after the logger closes the file; a later logger write would erase
        this header. Copies flight rows verbatim, streaming through a temporary
        file before replacing the original. A missing path creates metadata only.
        """
        header = self.csv_header()
        path = Path(PATH)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=path.parent, delete=False) as target:
                temporary = Path(target.name)
                if path.exists():
                    with path.open("r", encoding="utf-8", newline="") as source:
                        for line in source:
                            if line.startswith(self.CSV_PREFIX):
                                continue
                            if not line.startswith("#"):
                                target.write(header)
                                target.write(line)
                                shutil.copyfileobj(source, target)
                                break
                            target.write(line)
                        else:
                            target.write(header)
                else:
                    target.write(header)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def read_csv(self, PATH: str) -> bool:
        """Load leading metadata; old CSVs yield an empty course and False.

        Invalid metadata raises ValueError and leaves current geometry intact.
        """
        with open(PATH, "r", encoding="utf-8") as source:
            for line in source:
                if line.startswith(self.CSV_PREFIX):
                    payload = json.loads(line[len(self.CSV_PREFIX):])
                    if not isinstance(payload, dict) or payload.get("version") != 1:
                        raise ValueError("Unsupported obstacle course format")
                    records = payload.get("shapes")
                    if not isinstance(records, list):
                        raise ValueError("Obstacle course shapes must be a list")
                    try:
                        shapes = [self.Obstacle(**record) for record in records]
                    except (TypeError, KeyError) as error:
                        raise ValueError("Invalid obstacle geometry") from error
                    self.shapes = shapes
                    return True
                if not line.startswith("#"):
                    break
        self.clear()
        return False
