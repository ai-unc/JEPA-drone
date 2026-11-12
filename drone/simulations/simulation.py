from __future__ import annotations
import math
import asyncio

from sympy import Quaternion

from Subsystems.flight_controller import FlightController, Rotation
from drone.Subsystems.data.data import Action, State
from simulations.simulation_config import SimulationConfigs

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
        self.drone.set_pose(types.Transform({
            "translation": position,
            "rotation": orientation,
        }))

        self.state, self.time = self.read()
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
        self.local_data = True
        self.target_name = None

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

        self.drone = Drone(
            self.client,
            self.world,
            self.DEFAULT_DRONE_NAME,
        )

        if initial_state is not None:
            self.set_state(initial_state, time=0)

        self.state,self.time = self.read()
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
    @override
    def csv_header(self) -> str:
        if (self.configs is None):
            return ""
        return self.configs.config_to_csv() + "\n"
    
    @override
    def close(self) -> None:
        self.write_data_log()
        if self.client is not None:
            self.client.disconnect()