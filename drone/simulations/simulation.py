from __future__ import annotations

from Subsystems.flight_controller import FlightController, quaternion_to_rotation_matrix
from Subsystems.action import Action, Motor
from Subsystems.state import State
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, override

from projectairsim import Drone, ProjectAirSimClient, World


class ProjectAirSimSimulation(FlightController):

    MOTOR_TO_ACTUATOR = {
        Motor.FL: "Prop_FL_actuator",
        Motor.FR: "Prop_FR_actuator",
        Motor.BL: "Prop_RL_actuator",
        Motor.BR: "Prop_RR_actuator",
    }
    SCENE = "scene_basic_drone.jsonc"
    DEFAULT_DRONE_NAME = "Drone1"
    SIMULATION_FOLDER = Path(__file__).resolve().parent
    SIM_CONFIG_PATH = SIMULATION_FOLDER / "sim_config"
    LOAD_DELAY = 2
    @override
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.client: ProjectAirSimClient | None = None
        self.world: World | None = None
        self.drone: Drone | None = None

    @override
    def start(self) -> None:
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

    @override
    def read(self) -> tuple[State, np.double]:
        assert self.drone is not None

        kinematics = self.drone.get_ground_truth_kinematics()

        pose = kinematics["pose"]
        twist = kinematics["twist"]

        position = pose["position"]
        orientation = pose["orientation"]

        time_stamp= self.world.get_sim_time() / FlightController.NANO_SECOND

        state = State(
            rotation=quaternion_to_rotation_matrix(
                float(orientation["w"]),
                float(orientation["x"]),
                float(orientation["y"]),
                float(orientation["z"]),
            ),
            position=np.array([
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            ], dtype=float),
            velocity=np.array([
                float(twist["linear"]["x"]),
                float(twist["linear"]["y"]),
                float(twist["linear"]["z"]),
            ], dtype=float),
            angular_velocity=np.array([
                float(twist["angular"]["x"]),
                float(twist["angular"]["y"]),
                float(twist["angular"]["z"]),
            ], dtype=float)
        )

        return state,time_stamp
    @override
    def send(self, actions: Action) -> None:

        thrusts = actions.get_normalized()

        assert self.drone is not None

        control_signals = {}
        for motor_name, actuator_name in self.MOTOR_TO_ACTUATOR.items():
            control_signals[actuator_name] = float(thrusts[motor_name])

        self.drone.set_control_signals(control_signals)
    @override
    def step(
        self,
        action: Action | None = None) -> tuple[State, np.double]:

        assert self.world is not None

        if (action is not None):
            self.send(actions=action)
        self.world.continue_for_n_steps(
            FlightController.CONTROL_STEPS_PER_MILLISECOND,
            wait_until_complete=True,
        )
        state,time_stamp = self.read()
        if self.log_data:
            self.action_data.append(action)
            self.state_data.append(state)
            self.time_data.append(time_stamp)
        return state, time_stamp
    @override
    def close(self) -> None:
        self.write_data_log()
        if self.client is not None:
            self.client.disconnect()
