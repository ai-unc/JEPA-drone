from enum import Enum

import numpy as np
import csv
from Subsystems.action import Action, Motor
from Subsystems.state import State
def quaternion_to_rotation_matrix(w, x, y, z):
    norm = np.sqrt(w*w + x*x + y*y + z*z)

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - w*z),
            2*(x*z + w*y)
        ],
        [
            2*(x*y + w*z),
            1 - 2*(x*x + z*z),
            2*(y*z - w*x)
        ],
        [
            2*(x*z - w*y),
            2*(y*z + w*x),
            1 - 2*(x*x + y*y)
        ]
    ])


#Generic Flight Controller Template, assuming using pyserial or something similiar
class FlightController():
    CONTROL_HZ = 100.0
    CONTROL_PERIOD = np.double(1.0 / CONTROL_HZ)
    NANO_SECOND = 1_000_000_000
    MILLI_SECOND = 1_000_000
    CONTROL_NS = int(NANO_SECOND / CONTROL_HZ)
    CONTROL_STEPS_PER_MILLISECOND : int = CONTROL_NS // MILLI_SECOND
    DATA_PATH = "flight_logs/"
    def __init__(self) -> None:
        self.action_data = []
        self.state_data = []
        self.time_data = []
        self.log_data = False
        self.data_name = FlightController.DATA_PATH + "temp.csv"
    def start(self) -> None:
        pass
    #Prompts the user if they want to log data and what to name said data
    def log_prompt(self) -> None:
        self.log_data = input("Do you want to log data? (y/n): ").strip().lower() == "y"
        if (self.log_data):
            self.data_name = input("Name the data log file: (ex. temp.csv):\n")
            self.data_name = FlightController.DATA_PATH + self.data_name
    def send(self,action : Action) -> None:
        pass
    def read(self) -> tuple[State, float]:
        #Returns the current state and the timestamp.
        pass
    def step(self,action: Action | None = None) -> tuple[State, np.double]:
        #Steps the flight controller and returns the current state and timestamp.
        pass
    def close(self) -> None:
        pass
    def write_data_log(self):
        if not self.log_data:
            return
        """
        Log the collected time stamps, actions, and states into a csv.
        State:
            position: np.ndarray = np.zeros(3,dtype=float),
            velocity: np.ndarray = np.zeros(3,dtype=float),
            rotation: np.ndarray = np.zeros((3,3),dtype=float),
            angular_velocity: np.ndarray = np.zeros(3,dtype=float)
        Action:
             motor_thrusts: dict = {Motor.FL: 0, Motor.FR: 0, Motor.BL: 0, Motor.BR: 0},
        """
        header = [
            "time",
            "FL", "FR", "BL", "BR",
            "x", "y", "z",
            "vx", "vy", "vz",
            "r1x", "r1y", "r1z",
            "r2x", "r2y", "r2z",
            "r3x", "r3y", "r3z",
            "wx", "wy", "wz"
        ]

        with open(self.data_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for i in range(len(self.time_data)):
                time_stamp = self.time_data[i]
                action = self.action_data[i]
                state = self.state_data[i]

                writer.writerow([
                    time_stamp,

                    action.motor_thrusts[Motor.FL],
                    action.motor_thrusts[Motor.FR],
                    action.motor_thrusts[Motor.BL],
                    action.motor_thrusts[Motor.BR],

                    state.position[0],
                    state.position[1],
                    state.position[2],

                    state.velocity[0],
                    state.velocity[1],
                    state.velocity[2],

                    state.rotation[0, 0],
                    state.rotation[0, 1],
                    state.rotation[0, 2],

                    state.rotation[1, 0],
                    state.rotation[1, 1],
                    state.rotation[1, 2],

                    state.rotation[2, 0],
                    state.rotation[2, 1],
                    state.rotation[2, 2],

                    state.angular_velocity[0],
                    state.angular_velocity[1],
                    state.angular_velocity[2],
                ])

