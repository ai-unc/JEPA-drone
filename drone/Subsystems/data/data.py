from drone.simulations.simulation_config import SimulationConfigs

import os
import pandas as pd
import numpy as np
import random

class Action():
    ASSUMED_MAX_THRUST : np.double = 4.18  # Newtons
    FL = 0
    FR = 1
    BL = 2
    BR = 3
    @staticmethod
    def generate( motor_thrusts = np.array([
                    np.double(0.0),
                    np.double(0.0),
                    np.double(0.0),
                    np.double(0.0)
                ],dtype=np.double)) -> np.ndarray:
        for i in range(len(motor_thrusts)):
            if motor_thrusts[i] < 0:
                motor_thrusts[i] = 0
            if motor_thrusts[i] > Action.ASSUMED_MAX_THRUST:
                motor_thrusts[i] = Action.ASSUMED_MAX_THRUST
        return motor_thrusts
    @staticmethod
    def to_string(motor_thrusts) -> str:
        return (
            f"motor_thrusts=({motor_thrusts[Action.FL]:+7.3f}, {motor_thrusts[Action.FR]:+7.3f}, " +
            f"{motor_thrusts[Action.BL]:+7.3f}, {motor_thrusts[Action.BR]:+7.3f})"
        )
    @staticmethod
    def get_normalized(motor_thrusts) -> np.ndarray:
        thrusts = motor_thrusts.copy()
        for i in range(len(thrusts)):
            thrusts[i] = min(max(thrusts[i] / Action.ASSUMED_MAX_THRUST, 0.0), 1.0)
        return thrusts

class State():
    PX = 0
    PY = 1
    PZ = 2
    VX = 3
    VY = 4
    VZ = 5
    R1x = 6
    R1y = 7
    R1z = 8
    R2x = 9
    R2y = 10
    R2z = 11
    R3x = 12
    R3y = 13
    R3z = 14
    WX = 15
    WY = 16
    WZ = 17
    @staticmethod
    def generate(
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        rotation: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ) -> np.ndarray:
         # Safe handling of mutable default arguments
        pos = position if position is not None else np.zeros(3, dtype=np.double)
        vel = velocity if velocity is not None else np.zeros(3, dtype=np.double)
        rot = rotation if rotation is not None else np.eye(3, dtype=np.double) # Using Identity matrix for 3x3 rotation default
        ang_vel = angular_velocity if angular_velocity is not None else np.zeros(3, dtype=np.double)
        
        # Flatten and combine into a single 1D array of 18 doubles
        return np.concatenate([pos.ravel(), vel.ravel(), rot.ravel(), ang_vel.ravel()])

    @staticmethod
    def to_string(state: np.ndarray) -> str:
        return (
            f"position=({state[State.PX]:+7.3f}, {state[State.PY]:+7.3f}, {state[State.PZ]:+7.3f}) m \n" +
            f"velocity=({state[State.VX]:+7.3f}, {state[State.VY]:+7.3f}, {state[State.VZ]:+7.3f}) m/s \n " +
            f"rotation matrix=([\n" +
            f"  {state[State.R1x]:+7.3f}, {state[State.R1y]:+7.3f}, {state[State.R1z]:+7.3f}\n" +
            f"  {state[State.R2x]:+7.3f}, {state[State.R2y]:+7.3f}, {state[State.R2z]:+7.3f}\n" +
            f"  {state[State.R3x]:+7.3f}, {state[State.R3y]:+7.3f}, {state[State.R3z]:+7.3f}\n" +
            f"]) rad \n" +
            f"angular_velocity=({state[State.WX]:+7.3f}, {state[State.WY]:+7.3f}, {state[State.WZ]:+7.3f}) rad/s"
        )
    @staticmethod
    def get_position(state: np.ndarray) -> np.ndarray:
        return state[State.PX:State.PZ+1]

    @staticmethod
    def get_velocity(state: np.ndarray) -> np.ndarray:
        return state[State.VX:State.VZ+1]

    @staticmethod
    def get_rotation(state: np.ndarray) -> np.ndarray:
        return state[State.R1x:State.R3z+1].reshape(3, 3)

    @staticmethod
    def get_angular_velocity(state: np.ndarray) -> np.ndarray:
        return state[State.WX:State.WZ+1]


class DataSequence():
    """
    This class is used to store and manage a sequence of actions, states, and their corresponding time stamps for a drone simulation.
    -> During training this is useful for comparing SkyJEPA's predicted state and the actual state.
    -> Also used to make drones fly predefined action sequences.
    -> This is different from database.py as database is more so what the drone "knows" or has experienced, whereas DataSequence is a specific sequence of actions and states for simulation or training purposes.
    """
    CONTROL_FREQUENCY = 100
    DATA_SEQUENCE_FOLDER: str = "data_sequences"
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.states: list[np.ndarray] = []
        self.time_stamps: list[float] = []

    def add(self, action: Action, time_stamp: float, state: np.ndarray | None = None) -> None:
        self.actions.append(action)
        self.time_stamps.append(time_stamp)
        if state is not None:
            self.states.append(state)

    def to_string(self) -> str:
        return "\n".join(
            f"{time_stamp:+7.3f}: {Action.to_string(action)}"
            for action, time_stamp in zip(self.actions, self.time_stamps)
        )
    def get_action(self, time_stamp: float) -> np.ndarray | None:
        if not self.time_stamps:
            return None

        start_time = self.time_stamps[0]
        index = round((time_stamp - start_time) * DataSequence.CONTROL_FREQUENCY)

        if 0 <= index < len(self.actions):
            return self.actions[index]

        return None   
    #Be warned, a predefined sequence of actions does not mean the state is known.
    def get_state(self, time_stamp: float) -> np.ndarray | None:
        if not self.time_stamps:
            return None

        start_time = self.time_stamps[0]
        index = round((time_stamp - start_time) * DataSequence.CONTROL_FREQUENCY)

        if 0 <= index < len(self.states):
            return self.states[index]

        return None
    @staticmethod
    def generate_from_csv(
        file_name: str | None = None,
        start_time: float = 0,
    ) -> tuple["DataSequence", SimulationConfigs]:

        if file_name is None:
            file_name = input("Enter the action sequence file name (ex. gamepad_flight1.csv): \n")
        file_path = os.path.join(DataSequence.DATA_SEQUENCE_FOLDER, file_name)
        
        # Read simulation configuration from the first line
        with open(file_path, "r") as file:
            first_line = file.readline()

        if first_line.startswith(SimulationConfigs.CSV_PREFIX):
            sim_configs = SimulationConfigs.csv_to_config(first_line)
        else:
            # Backwards compatibility with CSVs that have no config header
            sim_configs = SimulationConfigs()

        # Ignore configuration metadata when reading flight data
        df = pd.read_csv(file_path, comment="#")

        sequence = DataSequence()

        dt = 1.0 / DataSequence.CONTROL_FREQUENCY

        for i, row in enumerate(df.itertuples(index=False)):

            action = Action.generate(
                motor_thrusts=np.array([
                    np.double(row.FL),
                    np.double(row.FR),
                    np.double(row.BL),
                    np.double(row.BR),
                ],dtype=np.double)
            )
            state = np.array([
                np.double(row.x),
                np.double(row.y),
                np.double(row.z),

                np.double(row.vx),
                np.double(row.vy),
                np.double(row.vz),

                np.double(row.r1x),
                np.double(row.r1y),
                np.double(row.r1z),
                np.double(row.r2x),
                np.double(row.r2y),
                np.double(row.r2z),
                np.double(row.r3x),
                np.double(row.r3y),
                np.double(row.r3z),

                np.double(row.wx),
                np.double(row.wy),
                np.double(row.wz)
            ],dtype=np.double)

            sequence.add(
                action,
                start_time + i * dt,
                state
            )
        return sequence, sim_configs
            
    @staticmethod
    def generate_random_sequence(
        time: float,
        frequency: float,
        start_time: float = 0,
        thrust_variance: tuple[float, float] = (0, Action.ASSUMED_MAX_THRUST)
    ) -> "DataSequence":

        if frequency <= 0:
            raise ValueError("frequency must be greater than 0")

        if frequency > DataSequence.CONTROL_FREQUENCY:
            raise ValueError(
                "frequency cannot be greater than CONTROL_FREQUENCY"
            )

        sequence = DataSequence()

        control_frequency = DataSequence.CONTROL_FREQUENCY
        steps = int(time * control_frequency)

        current_action: Action | None = None
        previous_change_index = -1

        for i in range(steps):

            # Which random-action interval are we currently in?
            change_index = int(i * frequency / control_frequency)

            # Generate a new action only when entering a new interval
            if change_index != previous_change_index:
                current_action = Action.generate(
                    motor_thrusts=np.array([
                        np.double(random.uniform(
                            thrust_variance[0], thrust_variance[1]
                        )),
                        np.double(random.uniform(
                            thrust_variance[0], thrust_variance[1]
                        )),
                        np.double(random.uniform(
                            thrust_variance[0], thrust_variance[1]
                        )),
                        np.double(random.uniform(
                            thrust_variance[0], thrust_variance[1]
                        )),
                    ],dtype=np.double)
                )

                previous_change_index = change_index

            assert current_action is not None

            time_stamp = start_time + (
                i / control_frequency
            )

            sequence.add(
                current_action,
                time_stamp,
                None
            )

        return sequence