from __future__ import annotations
import math

from sympy import Quaternion

from Subsystems.flight_controller import FlightController, Rotation
from Subsystems.action import Action
from Subsystems.state import State
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
    SIMULATION_FOLDER = Path(__file__).resolve().parent
    SIM_CONFIG_PATH = SIMULATION_FOLDER / "sim_config"
    LOAD_DELAY = 2

    SKY_POSE_Z = -80
    SPAWN_POSE_VARIATION = 100

    def set_pose_to_random_in_sky(self, height = SKY_POSE_Z):
        assert self.drone is not None
        import random
        position = {
            "x": random.uniform(-self.SPAWN_POSE_VARIATION, self.SPAWN_POSE_VARIATION),
            "y": random.uniform(-self.SPAWN_POSE_VARIATION, self.SPAWN_POSE_VARIATION),
            "z": height,
        }
        x, y, z, w = Rotation.random().as_quat()
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
    @override
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.client: ProjectAirSimClient | None = None
        self.world: World | None = None
        self.drone: Drone | None = None

    @override
    def start(self, initial_state: np.ndarray | None = None) -> None:
        self.client = ProjectAirSimClient()
        self.client.connect()


        self.world = World(
            self.client,
            self.SCENE,
            delay_after_load_sec=self.LOAD_DELAY,
            sim_config_path=str(self.SIM_CONFIG_PATH),
        )

        self.drone = Drone(
            self.client,
            self.world,
            self.DEFAULT_DRONE_NAME,
        )
        if initial_state is not None:
            self.set_state(initial_state, time=0)
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
    @override
    def read(self) -> tuple[np.ndarray, np.double]:
        assert self.drone is not None

        kinematics = self.drone.get_ground_truth_kinematics()

        pose = kinematics["pose"]
        twist = kinematics["twist"]

        position = pose["position"]
        orientation = pose["orientation"]

        time_stamp= self.world.get_sim_time() / FlightController.NANO_SECOND

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

        return state,time_stamp
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
        prev_state,prev_time_stamp = self.read()
        if (action is not None):
            self.send(actions=action)
        self.world.continue_for_n_steps(
            FlightController.CONTROL_STEPS_PER_MILLISECOND,
            wait_until_complete=True,
        )
        if self.log_data:
            self.action_data.append(action)
            self.state_data.append(prev_state)
            self.time_data.append(prev_time_stamp)
        return self.read()
    @override
    def close(self) -> None:
        self.write_data_log()
        if self.client is not None:
            self.client.disconnect()