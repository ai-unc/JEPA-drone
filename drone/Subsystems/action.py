import os

import pandas as pd
import math
import numpy as np
import random

class Motor():
    FL = "FL"
    FR = "FR"
    BL = "BL" #Backleft
    BR = "BR"

class Action():
    ASSUMED_MAX_THRUST : np.double = 4.18  # Newtons
    def __init__(
            self,
            motor_thrusts: dict = {Motor.FL: np.double(0), Motor.FR: np.double(0), Motor.BL: np.double(0), Motor.BR: np.double(0)},
            ):
        self.motor_thrusts = motor_thrusts #Newtons
        for motor in self.motor_thrusts:
            if self.motor_thrusts[motor] < 0:
                self.motor_thrusts[motor] = 0
            if self.motor_thrusts[motor] > Action.ASSUMED_MAX_THRUST:
                self.motor_thrusts[motor] = Action.ASSUMED_MAX_THRUST
    def to_string(self):
        return (
            f"motor_thrusts=({self.motor_thrusts[Motor.FL]:+7.3f}, {self.motor_thrusts[Motor.FR]:+7.3f}, " +
            f"{self.motor_thrusts[Motor.BL]:+7.3f}, {self.motor_thrusts[Motor.BR]:+7.3f})"
        )
    def get_normalized(self) ->None:
        thrusts = self.motor_thrusts.copy()
        for motor in self.motor_thrusts:
            thrusts[motor] = min(max(thrusts[motor] / Action.ASSUMED_MAX_THRUST, 0.0), 1.0)
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
    def get_action(self, time_stamp: float) -> Action | None:
        if not self.time_stamps:
            return None

        start_time = self.time_stamps[0]
        index = round((time_stamp - start_time) * ActionSequence.CONTROL_FREQUENCY)

        if 0 <= index < len(self.actions):
            return self.actions[index]

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

            action = Action(
                motor_thrusts={
                    Motor.FL: float(row.FL),
                    Motor.FR: float(row.FR),
                    Motor.BL: float(row.BL),
                    Motor.BR: float(row.BR),
                }
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
                current_action = Action(
                    motor_thrusts={
                        Motor.FL: random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        ),
                        Motor.FR: random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        ),
                        Motor.BL: random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        ),
                        Motor.BR: random.uniform(
                            0, Action.ASSUMED_MAX_THRUST
                        ),
                    }
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