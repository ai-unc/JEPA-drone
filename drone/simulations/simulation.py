from __future__ import annotations
import math
import asyncio
import time
import copy

from sympy import Quaternion

from Subsystems.flight_controller import FlightController, Rotation
from drone.Subsystems.data.data import Action, State
from simulations.simulation_config import SimulationConfigs
from simulations.obstacle_course import ObstacleCourse
from simulations.APF import APF
from simulations.SimCam import SimCamera

import numpy as np
from pathlib import Path
from typing import override

from projectairsim import Drone, ProjectAirSimClient, World, types


class ProjectAirSimSimulation(FlightController):

    MOTOR_TO_ACTUATOR = {
        Action.FL: "Prop_FL_actuator",
        Action.FR: "Prop_FR_actuator",
        Action.BL: "Prop_RL_actuator",
        Action.BR: "Prop_RR_actuator",
    }
    SCENE = "scene_basic_drone.jsonc"
    DEFAULT_DRONE_NAME = "Drone1"
    LOAD_DELAY = 0.1

    SKY_POSE_Z = -80
    SPAWN_POSE_VARIATION = 100
    SPAWN_HEIGHT_VARIATION = 50

    TARGET_POSE_VARIATION = [[-100,100],[-100,100],[-30,30]]
    TARGET_NAME="FlightTarget"
    TARGET_ASSET="Sphere"
    TARGET_SIZE = [8,8,8]
    MIN_DISTANCE_FROM_TARGET = 17

    def target_reached(self,target_position : np.ndarray, current_position : np.ndarray):
        x,y,z = current_position
        tx,ty,tz = target_position
        distance_from_target = State.get_distance(x,tx,y,ty,z,tz)
        return distance_from_target < self.MIN_DISTANCE_FROM_TARGET
    
    def generate_target(self,reference_pose : np.ndarray) -> np.ndarray:
        """
        Input a reference position for the target generation
        Returns target position
        """
        rng = np.random.default_rng()
        target_pose = types.Transform({
            "translation": {
                "x": float(
                    reference_pose[State.PX] + rng.uniform(self.TARGET_POSE_VARIATION[State.PX][0],self.TARGET_POSE_VARIATION[State.PX][1])
                ),
                "y": float(
                    reference_pose[State.PY] + rng.uniform(self.TARGET_POSE_VARIATION[State.PY][0],self.TARGET_POSE_VARIATION[State.PY][1])
                ),
                "z": float(min(
                    self.SKY_POSE_Z,
                    reference_pose[State.PZ] + rng.uniform(self.TARGET_POSE_VARIATION[State.PZ][0],self.TARGET_POSE_VARIATION[State.PZ][1]),
                )),
            },
            # Identity quaternion: no rotation.
            "rotation": {
                "w": 1.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
        })
        if self.obstacle_course is not None:
            goal_shape = next((shape for shape in self.obstacle_course.shapes
                               if shape.shape == ObstacleCourse.Shape.GOAL), None)
            if goal_shape is not None:
                target_pose["translation"] = dict(zip(("x", "y", "z"), goal_shape.position.tolist()))
            else:
                target_pose["translation"] = dict(zip(("x", "y", "z"), self.obstacle_course.goal.tolist()))
        if (self.configs.render):
            if self.target_name is None:
                self.target_name = self.world.spawn_object(
                    object_name=self.TARGET_NAME,
                    asset_path=self.TARGET_ASSET,
                    object_pose=target_pose,
                    object_scale=self.TARGET_SIZE,
                    enable_physics=False,
                )
                if not self.target_name:
                    raise RuntimeError("Failed to spawn FlightTarget.")
                if self.vision_enabled:
                    self.world.set_segmentation_id_by_name(self.target_name, 3, False, True)
            else:
                self.world.set_object_pose(
                    object_name=self.target_name,
                    object_pose=target_pose,
                    teleport=True,
                )
        target_position = np.array([
            target_pose["translation"]["x"],
            target_pose["translation"]["y"],
            target_pose["translation"]["z"],
        ], dtype=float)
        return target_position


    def set_pose_to_random_in_sky(self, height = SKY_POSE_Z, rnd_pitch_and_roll = True) -> tuple[np.ndarray,dict]:
        assert self.drone is not None
        import random
        
        if self.obstacle_course is not None:
            self.obstacle_course.validate_drone_spawn()
            position = dict(zip(("x", "y", "z"), self.obstacle_course.drone_spawn.tolist()))
            direction = self.obstacle_course.goal - self.obstacle_course.drone_spawn
            x, y, z, w = Rotation.from_euler("z", np.arctan2(direction[1], direction[0])).as_quat()
        else:
            position = {
                        "x": random.uniform(-self.SPAWN_POSE_VARIATION, self.SPAWN_POSE_VARIATION),
                        "y": random.uniform(-self.SPAWN_POSE_VARIATION, self.SPAWN_POSE_VARIATION),
                        "z": random.uniform(height - self.SPAWN_HEIGHT_VARIATION, height + self.SPAWN_HEIGHT_VARIATION),
                    }
            x, y, z, w = Rotation.random().as_quat()
            if not rnd_pitch_and_roll:
                # Start level while preserving the sampled heading.
                yaw = Rotation.from_quat([x, y, z, w]).as_euler("xyz")[2]
                x, y, z, w = Rotation.from_euler("z", yaw).as_quat()
       
        orientation = {
            "w": float(w),
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }
        pose_set = self.drone.set_pose(types.Transform({
            "translation": position,
            "rotation": orientation,
        }))

        self.state, self.time = self.read()
        if self.obstacle_course is not None:
            actual_position = State.get_position(self.state)
            if not pose_set or not np.allclose(actual_position, self.obstacle_course.drone_spawn, atol=0.1, rtol=0):
                raise RuntimeError(f"Course spawn failed: requested {self.obstacle_course.drone_spawn}, got {actual_position}")
            self.obstacle_course.validate_drone_spawn(actual_position)
            # SetPose updates the physics state before Unreal moves its mesh.
            # Wait for the rendered drone, which can be stopped by a sweep.
            deadline = time.monotonic() + 3.0
            while True:
                bounds = self.world.get_3d_bounding_box(
                    self.DEFAULT_DRONE_NAME, types.BoxAlignment.WORLD_AXIS
                )
                rendered = np.array([bounds["center"][k] for k in ("x", "y", "z")])
                if np.linalg.norm(rendered - actual_position) < 1.0:
                    self.obstacle_course.validate_drone_spawn(rendered)
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Rendered drone did not reach course spawn: requested {actual_position}, "
                        f"rendered {rendered}. Movement may have been blocked by a collision."
                    )
                time.sleep(0.02)
        return np.array([position["x"], position["y"], position["z"]], dtype=float), orientation
    def config_prompt(self) -> None:
        #CONFIGS MUST BE SET BEFORE THE SIMULATION STARTS!!!
        user_input = input("Random Configs? (y/n): ")
        if user_input.lower() == "y":
            self.configs.randomize()
        self.configs.write()

    def set_configs(self, configs: SimulationConfigs) -> None:
        self.configs = configs
        self.configs.write()

    @override
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.client: ProjectAirSimClient | None = None
        self.world: World | None = None
        self.drone: Drone | None = None
        self.configs = SimulationConfigs()
        self.obstacle_course : ObstacleCourse | None  = None
        self.apf = APF()
        self.local_data = True
        self.target_name = None
        self.camera = SimCamera()
        self.vision_enabled = False

    @override
    def start(self, initial_state: np.ndarray | None = None) -> None:
        self.client = ProjectAirSimClient()
        self.client.connect()

        self.world = World(
            self.client,
            self.SCENE,
            delay_after_load_sec=self.LOAD_DELAY,
            sim_config_path=str(self.configs.path),
        )

        if self.obstacle_course is not None:
            spawn_position = (self.obstacle_course.drone_spawn if initial_state is None
                              else State.get_position(initial_state))
            self.obstacle_course.validate_drone_spawn(spawn_position)
            # FastPhysics can report SetPose while Unreal still displays the
            # initial scene origin. Create the robot at the course spawn so
            # no paused teleport or collision sweep is needed to get there.
            scene = copy.deepcopy(self.world.get_configuration())
            direction = self.obstacle_course.goal - spawn_position
            yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
            actor = next(a for a in scene["actors"] if a["name"] == self.DEFAULT_DRONE_NAME)
            actor["origin"] = {
                "xyz": " ".join(str(float(v)) for v in spawn_position),
                "rpy-deg": f"0 0 {yaw}",
            }
            self.world.load_scene(scene, delay_after_load_sec=self.LOAD_DELAY)

        self.drone = Drone(
            self.client,
            self.world,
            self.DEFAULT_DRONE_NAME,
        )

        if initial_state is not None:
            self.set_state(initial_state, time=0)

        self.state,self.time = self.read()

        if (self.obstacle_course is not None):
            # Move out of the obstacle region before creating collision meshes.
            if initial_state is None:
                self.set_pose_to_random_in_sky(rnd_pitch_and_roll=False)
            self.generate_obstacle_course()
        if self.vision_enabled:
            # Baseplate ground and all default environment geometry are walls.
            self.world.set_segmentation_id_by_name(".*", 1, True, True)
            self.world.set_segmentation_id_by_name("obstacle_.*", 2, True, True)
            self.world.set_segmentation_id_by_name("FlightTarget.*|goal_.*", 3, True, True)
            self.world.set_segmentation_id_by_name("Drone1.*|ceiling_light_.*", 0, True, True)
            self.camera.attach(self)
    
    @override
    def set_state(self, state: np.ndarray,time : np.double = 0) -> None:
        rotation_matrix = state[State.R1x:State.R3z+1].reshape(3,3)
        x, y, z, w = Rotation.from_matrix(rotation_matrix).as_quat()
        self.drone.set_ground_truth_kinematics(
            {
                "time_stamp" : time,
                "pose" : {
                    "position": types.Vector3({
                        "x" : state[State.PX],
                        "y" : state[State.PY],
                        "z" : state[State.PZ],
                    }
                    ),
                    "orientation": types.Quaternion({
                        "w": w,
                        "x": x,
                        "y": y,
                        "z": z,
                    })
                },
                "twist" : {
                    "linear": types.Vector3({
                        "x" : state[State.VX],
                        "y" : state[State.VY],
                        "z" : state[State.VZ],
                    }
                    ),
                    "angular": types.Vector3({
                        "x" : state[State.WX],
                        "y" : state[State.WY],
                        "z" : state[State.WZ],
                    })
                },
                "accels" : {
                    "linear": types.Vector3({
                        "x" : 0,
                        "y" : 0,
                        "z" : 0,
                    }),
                    "angular": types.Vector3({
                        "x" : 0,
                        "y" : 0,
                        "z" : 0,
                    })
                }
            }
            
        )

        self.state, self.time = self.read()
    @override
    def read(self) -> tuple[np.ndarray, np.double]:
        assert self.drone is not None

        kinematics = self.drone.get_ground_truth_kinematics()

        time_stamp = self.world.get_sim_time() / FlightController.NANO_SECOND
        return self._state_from_kinematics(kinematics), time_stamp

    @staticmethod
    def _state_from_kinematics(kinematics: dict) -> np.ndarray:
        """Use the same state layout for direct reads and bundled step results."""
        pose = kinematics["pose"]
        twist = kinematics["twist"]
        position = pose["position"]
        orientation = pose["orientation"]

        state = State.generate(
            position=np.array([
                np.double(position["x"]),
                np.double(position["y"]),
                np.double(position["z"]),
            ]),
            rotation=Rotation.from_quat([
                np.double(orientation["w"]),
                np.double(orientation["x"]),
                np.double(orientation["y"]),
                np.double(orientation["z"])
            ], scalar_first=True).as_matrix(),
            velocity=np.array([
                np.double(twist["linear"]["x"]),
                np.double(twist["linear"]["y"]),
                np.double(twist["linear"]["z"]),
            ]),
            angular_velocity=np.array([
                np.double(twist["angular"]["x"]),
                np.double(twist["angular"]["y"]),
                np.double(twist["angular"]["z"]),
            ])
        )

        return state
    @override
    def send(self, actions: np.ndarray) -> None:

        thrusts = Action.get_normalized(actions)

        assert self.drone is not None

        control_signals = {}
        for motor_name, actuator_name in self.MOTOR_TO_ACTUATOR.items():
            control_signals[actuator_name] = thrusts[motor_name]

        self.drone.set_control_signals(control_signals)
    @override
    def step(
        self,
        action: np.ndarray | None = None) -> tuple[np.ndarray, np.double]:

        assert self.world is not None
        prev_state,prev_time_stamp = self.state,self.time
        if (action is not None):
            self.send(actions=action)
        result = self.world.step(dt_ns=FlightController.CONTROL_NS)
        robot_state = result["robots"][self.DEFAULT_DRONE_NAME]["state"]
        next_state = self._state_from_kinematics({
            "pose": {
                "position": robot_state["position"],
                "orientation": robot_state["orientation"],
            },
            "twist": {
                "linear": robot_state["linear_velocity"],
                "angular": robot_state["angular_velocity"],
            },
        })
        next_time = result["sim_time_ns"] / FlightController.NANO_SECOND

        if self.log_data:
            self.action_data.append(action)
            self.state_data.append(prev_state)
            self.time_data.append(prev_time_stamp)
        if (self.local_data):
            self.database.update_history(prev_state, action, prev_time_stamp)
        self.state, self.time = next_state, next_time
        return self.state,self.time

    def obstacle_avoidance_velocity(self) -> tuple[np.ndarray, float]:
        """Target world velocity and travel heading from known course geometry."""
        if self.obstacle_course is None:
            raise RuntimeError("Obstacle avoidance requires an obstacle course")
        velocity = self.apf.run(State.get_position(self.state), self.obstacle_course)
        return velocity, self.apf.yaw

    def generate_obstacle_course(self) -> None:
        """Spawn the attached course's stored geometry into the current world."""
        if self.obstacle_course is None:
            return
        if self.world is None:
            raise RuntimeError("Start the simulation before generating obstacles.")

        self.generate_target(self.obstacle_course.goal)
        for index, obstacle in enumerate(self.obstacle_course.shapes):
            if obstacle.shape == ObstacleCourse.Shape.GOAL:
                continue  # FlightTarget is spawned by generate_target above.
            x, y, z, w = Rotation.from_euler(
                "xyz", obstacle.rotation, degrees=True
            ).as_quat()
            pose = types.Transform({
                "translation": dict(zip(("x", "y", "z"), obstacle.position.tolist())),
                "rotation": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)},
            })
            name = self.world.spawn_object(
                object_name=f"{obstacle.name}_{index}",
                asset_path=obstacle.shape.value,
                object_pose=types.Transform({
                    "translation": {"x": 0., "y": 0., "z": 0.},
                    "rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                }),
                object_scale=[1., 1., 1.],
                enable_physics=False,
            )
            if not name:
                raise RuntimeError(f"Failed to spawn obstacle {index} ({obstacle.shape.value}).")
            # Scale is a mesh multiplier, not a dimension in metres. Measure
            # the actual asset at identity, including any off-centre pivot.
            bounds = self.world.get_3d_bounding_box(name, types.BoxAlignment.WORLD_AXIS)
            native_size = np.array([bounds["size"][k] for k in ("x", "y", "z")])
            native_center = np.array([bounds["center"][k] for k in ("x", "y", "z")])
            if not np.all(np.isfinite(native_size)) or np.any(native_size <= 0):
                raise RuntimeError(f"Invalid mesh bounds for {name}: {bounds}")
            scale = obstacle.size / native_size
            self.world.set_object_scale(name, scale.tolist())
            rotation = Rotation.from_euler("xyz", obstacle.rotation, degrees=True)
            origin = obstacle.position - rotation.apply(native_center * scale)
            pose["translation"] = dict(zip(("x", "y", "z"), origin.tolist()))
            self.world.set_object_pose(name, pose, teleport=True)
            actual = self.world.get_3d_bounding_box(name, types.BoxAlignment.WORLD_AXIS)
            center = np.array([actual["center"][k] for k in ("x", "y", "z")])
            extent = np.array([actual["size"][k] for k in ("x", "y", "z")]) / 2
            if (not np.allclose(center, obstacle.position, atol=0.05, rtol=0)
                    or np.any(extent > obstacle.bounding_extent() + 0.05)):
                raise RuntimeError(f"Rendered geometry does not match course record for {name}: {actual}")

        # Four quadrant lights and one central light hang below the ceiling.
        # Lights are scene fixtures, so they are not part of the course log.
        ceiling = self.obstacle_course.shapes[4]
        ceiling_bottom = ceiling.position[2] + ceiling.size[2] / 2
        x_edges = (self.obstacle_course.min_x - self.obstacle_course.spawn_radius,
                   self.obstacle_course.max_x + self.obstacle_course.spawn_radius)
        y_edges = (self.obstacle_course.min_y - self.obstacle_course.spawn_radius,
                   self.obstacle_course.max_y + self.obstacle_course.spawn_radius)
        x_mid = (x_edges[0] + x_edges[1]) / 2
        y_mid = (y_edges[0] + y_edges[1]) / 2
        x_quarters = ((x_edges[0] + x_mid) / 2, (x_mid + x_edges[1]) / 2)
        y_quarters = ((y_edges[0] + y_mid) / 2, (y_mid + y_edges[1]) / 2)
        light_positions = [(x, y) for y in y_quarters for x in x_quarters]
        light_positions.append((x_mid, y_mid))
        for index, (x, y) in enumerate(light_positions):
                name = self.world.spawn_object(
                    object_name=f"ceiling_light_{index}",
                    asset_path="PointLightActor",
                    object_pose=types.Transform({
                        "translation": {"x": float(x), "y": float(y),
                                        "z": float(ceiling_bottom + 2)},
                        "rotation": {"w": 1., "x": 0., "y": 0., "z": 0.},
                    }),
                    object_scale=[1., 1., 1.],
                    enable_physics=False,
                )
                if not name:
                    raise RuntimeError(f"Failed to spawn ceiling light {index}")
                # Unreal's SetAttenuationRadius takes centimetres, despite the
                # Python API describing this value as metres.
                if not self.world.set_light_object_radius(name, 55000000.):
                    raise RuntimeError(f"Failed to set radius for {name}")
                if not self.world.set_light_object_intensity(name, 5000000.):
                    raise RuntimeError(f"Failed to set intensity for {name}")
    
    @override
    def csv_header(self) -> str:
        header = self.configs.config_to_csv() + "\n" if self.configs is not None else ""
        if self.obstacle_course is not None:
            header += self.obstacle_course.csv_header()
        return header
    
    @override
    def close(self) -> None:
        self.camera.close()
        self.write_data_log()
        if self.client is not None:
            self.client.disconnect()
