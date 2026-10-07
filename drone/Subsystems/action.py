import os

import pandas as pd
import math
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

class ActionSequence():
    CONTROL_FREQUENCY = 100
    ACTION_SEQUENCE_FOLDER: str = "action_sequences"
    def __init__(self) -> None:
        self.actions: list[Action] = []
        self.time_stamps: list[float] = []

    def append(self, action: Action, time_stamp: float) -> None:
        self.actions.append(action)
        self.time_stamps.append(time_stamp)

    def to_string(self) -> str:
        return "\n".join(
            f"{time_stamp:+7.3f}: {action.to_string()}"
            for action, time_stamp in zip(self.actions, self.time_stamps)
        )
    def get_action(self, time_stamp: float) -> np.ndarray | None:
        if not self.time_stamps:
            return None

        start_time = self.time_stamps[0]
        index = round((time_stamp - start_time) * ActionSequence.CONTROL_FREQUENCY)

        if 0 <= index < len(self.actions):
            return self.actions[index].copy()

        return None

    @staticmethod
    def generate_from_csv(
        file_name: str | None = None,
        start_time: float = 0,
    ) -> "ActionSequence":

        if file_name is None:
            file_name = input("Enter the action sequence file name (ex. gamepad_flight1.csv): \n")
        file_path = os.path.join(ActionSequence.ACTION_SEQUENCE_FOLDER, file_name)
        df = pd.read_csv(file_path)

        sequence = ActionSequence()

        dt = 1.0 / ActionSequence.CONTROL_FREQUENCY

        for i, row in enumerate(df.itertuples(index=False)):

            action = Action.generate(
                motor_thrusts=np.array([
                    np.double(row.FL),
                    np.double(row.FR),
                    np.double(row.BL),
                    np.double(row.BR),
                ],dtype=np.double)
            )

            sequence.append(
                action,
                start_time + i * dt
            )

        return sequence

    @staticmethod
    def generate_random_sequence(
        time: float,
        frequency: float,
        start_time: float = 0
    ) -> "ActionSequence":

        if frequency <= 0:
            raise ValueError("frequency must be greater than 0")

        if frequency > ActionSequence.CONTROL_FREQUENCY:
            raise ValueError(
                "frequency cannot be greater than CONTROL_FREQUENCY"
            )

        sequence = ActionSequence()

        control_frequency = ActionSequence.CONTROL_FREQUENCY
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
                            0, Action.ASSUMED_MAX_THRUST
                        )),
                        np.double(random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        )),
                        np.double(random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        )),
                        np.double(random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        )),
                    ],dtype=np.double)
                )

                previous_change_index = change_index

            assert current_action is not None

            time_stamp = start_time + (
                i / control_frequency
            )

            sequence.append(
                current_action,
                time_stamp
            )

        return sequence