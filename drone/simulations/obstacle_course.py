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
from scipy.spatial.transform import Rotation


class ObstacleCourse:
    class Shape(Enum):
        RECTANGLE = "Cube"
        SPHERE = "Sphere"
        DISK = "Cylinder"
        GOAL = "Cone"
    class ObstacleType(Enum):
        obstacle = "obstacle"
        wall = "wall"

    @dataclass
    class Obstacle:
        shape: "ObstacleCourse.Shape"
        size: np.ndarray
        position: np.ndarray
        rotation: np.ndarray
        name: str = "obstacle"

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
            return dict(name=self.name, shape=self.shape.value, size=self.size.tolist(),
                        position=self.position.tolist(), rotation=self.rotation.tolist())

        def to_string(self) -> str:
            return json.dumps(self.to_dict(), allow_nan=False)

        def bounding_extent(self) -> np.ndarray:
            """Conservative world-axis half extents, including rotation."""
            return np.abs(Rotation.from_euler(
                "xyz", self.rotation, degrees=True
            ).as_matrix()) @ (self.size / 2)

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
        self.min_size, self.max_size = 10, 20
        self.rectangles = 20
        self.spheres = 10
        self.disks = 10

        self.walls = 4
        self.wall_width = 20
        self.room_height = 12
        self.drone_clearance = 2

        self.goal = np.array([0., 0., -5.])
        self.goal_variation = np.array([50., 50., 0.])
        self.goal_radius = 50 #No obstacles around the goal for this many meters

        # Expand the walls beyond the obstacle region by this clear corridor.
        self.spawn_radius = 40
        self.spawn_wall_distance = 10

        self.drone_spawn = np.array([0., 0., -5.])


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
        low = np.array([self.min_x, self.min_y], dtype=float) - self.spawn_radius
        high = np.array([self.max_x, self.max_y], dtype=float) + self.spawn_radius
        if (self.wall_width <= 0 or self.room_height <= 1
                or self.spawn_radius <= 2 * self.drone_clearance
                or not self.drone_clearance < self.spawn_wall_distance < self.spawn_radius - self.drone_clearance
                or np.any(high - low <= 2 * self.spawn_radius + self.max_size * 2)):
            raise ValueError("Room is too small for obstacles and a safe spawn corridor")
        self.goal = np.array([0., 0., -1.]) + rng.uniform(-self.goal_variation, self.goal_variation)
        # Choose a point in the clear perimeter strip, inset from both faces.
        self.drone_spawn = np.r_[rng.uniform(low + self.drone_clearance,
                                            high - self.drone_clearance), -10.]
        axis = int(rng.integers(2))
        inset = self.spawn_wall_distance
        self.drone_spawn[axis] = (low[axis] + inset if rng.integers(2) else high[axis] - inset)
        shapes = []
        # Bounds describe the inner wall faces; adjoining slabs overlap.
        width, depth = high - low
        cx, cy = (low + high) / 2
        t, h, g = self.wall_width, self.room_height, self.ground_level
        for size, position in (
            ([t, depth + 2*t, h + t], [low[0]-t/2, cy, g-h/2]),
            ([t, depth + 2*t, h + t], [high[0]+t/2, cy, g-h/2]),
            ([width + 2*t, t, h + t], [cx, low[1]-t/2, g-h/2]),
            ([width + 2*t, t, h + t], [cx, high[1]+t/2, g-h/2]),
            ([width + 2*t, depth + 2*t, t], [cx, cy, g-h-t/2]),
        ):
            shapes.append(self.Obstacle(self.Shape.RECTANGLE, size, position, [0,0,0], name="wall"))
        for kind, count in zip((self.Shape.RECTANGLE, self.Shape.SPHERE, self.Shape.DISK), counts):
            for _ in range(count):
                size = rng.uniform(self.min_size, self.max_size, 3)
                if kind == self.Shape.SPHERE:
                    size[:] = size[0]
                elif kind == self.Shape.DISK:
                    size[1] = size[0]
                rotation = rng.uniform(self.min_angles, self.max_angles, 3)
                obstacle = self.Obstacle(kind, size, [0, 0, self.ground_level], rotation)
                # Rotated bounding boxes conservatively protect the entire geometry.
                extent = obstacle.bounding_extent()
                lower = low + self.spawn_radius + extent[:2]
                upper = high - self.spawn_radius - extent[:2]
                for attempt in range(10000):
                    xy = rng.uniform(lower, upper)
                    distance = np.linalg.norm(np.maximum(np.abs(xy - self.goal[:2]) - extent[:2], 0))
                    if distance >= self.goal_radius:
                        break
                else:
                    raise ValueError("Cannot place obstacles outside goal clearance")
                obstacle.position[:] = [*xy, rng.uniform(g - obstacle.relative_height()/2, g)]
                shapes.append(obstacle)
        shapes.append(self.Obstacle(self.Shape.GOAL, [8, 8, 8], self.goal,
                                    [0, 0, 0], name="goal"))
        self.shapes = shapes
        self.validate_drone_spawn()

    def validate_drone_spawn(self, position=None) -> None:
        """Reject a spawn whose drone clearance intersects any stored geometry."""
        position = self.drone_spawn if position is None else np.asarray(position, dtype=float)
        if not np.all(np.isfinite(position)) or position.shape != (3,):
            raise ValueError("Drone spawn must contain three finite coordinates")
        if position[2] + self.drone_clearance >= self.ground_level:
            raise ValueError("Drone spawn is too close to the ground")
        walls = [s for s in self.shapes if s.name == "wall"]
        if len(walls) == 5:
            # Derive inner faces from serialized geometry, including on replay.
            xwalls = sorted(walls[:4], key=lambda s: s.position[0])
            ywalls = sorted(walls[:4], key=lambda s: s.position[1])
            low = np.array([xwalls[0].position[0] + xwalls[0].bounding_extent()[0],
                            ywalls[0].position[1] + ywalls[0].bounding_extent()[1]])
            high = np.array([xwalls[-1].position[0] - xwalls[-1].bounding_extent()[0],
                             ywalls[-1].position[1] - ywalls[-1].bounding_extent()[1]])
            distances = np.r_[position[:2] - low, high - position[:2]]
            if np.any(distances < self.drone_clearance) or distances.min() > self.spawn_wall_distance + 0.1:
                raise ValueError("Drone spawn must be inside the room near a wall")
            for obstacle in self.shapes:
                if obstacle.name != "obstacle":
                    continue
                extent = obstacle.bounding_extent()[:2]
                if (np.any(obstacle.position[:2] - extent < low + self.spawn_radius - 1e-6)
                        or np.any(obstacle.position[:2] + extent > high - self.spawn_radius + 1e-6)):
                    raise ValueError("Obstacle geometry intrudes into the wall spawn corridor")
        for obstacle in self.shapes:
            distance = np.linalg.norm(np.maximum(
                np.abs(position - obstacle.position) - obstacle.bounding_extent(), 0
            ))
            if distance < self.drone_clearance:
                raise ValueError(f"Drone spawn intersects {obstacle.name} clearance")

    def clear(self) -> None:
        """Clear stored geometry; world objects must be removed by simulation code."""
        self.shapes.clear()

    def to_string(self) -> str:
        """Serialize exact geometry for replay."""
        return json.dumps({"version": 1, "goal": self.goal.tolist(),
                          "drone_spawn": self.drone_spawn.tolist(),
                          "shapes": [s.to_dict() for s in self.shapes]},
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
                    goal_shapes = [s for s in shapes if s.shape == self.Shape.GOAL]
                    if len(goal_shapes) > 1:
                        raise ValueError("Obstacle course must have at most one goal")
                    self.goal = (goal_shapes[0].position.copy() if goal_shapes else
                                 np.array(payload.get("goal", [0,0,-10]), dtype=float))
                    if not goal_shapes:
                        shapes.append(self.Obstacle(self.Shape.GOAL, [8, 8, 8],
                                                    self.goal, [0, 0, 0], name="goal"))
                    self.shapes = shapes
                    self.drone_spawn = np.array(payload.get("drone_spawn", [0,0,-10]), dtype=float)
                    return True
                if not line.startswith("#"):
                    break
        self.clear()
        return False
