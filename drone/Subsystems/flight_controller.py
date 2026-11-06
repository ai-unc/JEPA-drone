from enum import Enum

import numpy as np
import csv
from scipy.spatial.transform import Rotation
from drone.Subsystems.data.data import Action
from drone.Subsystems.data.data import State
from drone.Subsystems.data.database import Database



#Generic Flight Controller Template, assuming using pyserial or something similiar
class FlightController():
    CONTROL_HZ = 100.0
    CONTROL_PERIOD = np.double(1.0 / CONTROL_HZ)
    NANO_SECOND = 1_000_000_000
    MILLI_SECOND = 1_000_000
    CONTROL_NS = int(NANO_SECOND / CONTROL_HZ)
    CONTROL_STEPS_PER_MILLISECOND : int = CONTROL_NS // MILLI_SECOND
    DATA_PATH = "flight_logs/"
    FLIGHT_TIME = 10 #Seconds
    def __init__(self) -> None:
        self.action_data : list[np.ndarray] = []
        self.state_data : list[np.ndarray] = []
        self.time_data = []
        self.database = Database()
        self.log_data = False
        self.data_name = self.DATA_PATH + "temp.csv"
        self.state = None
        self.time = None
        self.split_logs = False
        self.split_time = self.FLIGHT_TIME
    def start(self,initial_state : np.ndarray | None = None) -> None:
        pass
    #Prompts the user if they want to log data and what to name said data
    def log_prompt(self) -> None:
        self.log_data = input("Do you want to log data? (y/n): ").strip().lower() == "y"
        if (self.log_data):
            self.data_name = input("Name the csv data log file: (ex. temp):\n")
            self.data_name = self.DATA_PATH + self.data_name
    def send(self,action : np.ndarray) -> None:
        pass
    def read(self) -> tuple[State, float]:
        #Returns the current state and the timestamp.
        pass
    def step(self,action: np.ndarray | None = None) -> tuple[State, np.double]:
        #Steps the flight controller and returns the current state and timestamp.
        #Also Update the database
        pass
    def close(self) -> None:
        pass
    def set_state(self, state: np.ndarray) -> None:
        pass
    def csv_header(self) -> str:
        return ""
    def write_data_log(self):
        if not self.log_data:
            return
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
        if self.split_logs:
            samples = self.split_time * self.CONTROL_HZ
            if not np.isfinite(samples) or samples < 1 or not float(samples).is_integer():
                raise ValueError("split_time must represent a positive whole number of control samples")
            samples_per_flight = int(samples)
            # Only write complete subflights; keep original timestamps and row order.
            flights = [
                (f"{self.data_name}_{flight + 1}.csv", start, start + samples_per_flight)
                for flight, start in enumerate(
                    range(0, len(self.time_data) - samples_per_flight + 1, samples_per_flight)
                )
            ]
        else:
            flights = [(self.data_name + ".csv", 0, len(self.time_data))]

        for filename, start, stop in flights:
            with open(filename, "w", newline="") as file:
                file.write(self.csv_header())
            
                writer = csv.writer(file)
                writer.writerow(header)

                for i in range(start, stop):
                    time_stamp = self.time_data[i]
                    action = self.action_data[i]
                    state = self.state_data[i]

                    writer.writerow([
                        time_stamp,

                        action[Action.FL],
                        action[Action.FR],
                        action[Action.BL],
                        action[Action.BR],

                        state[State.PX], #Position X
                        state[State.PY],
                        state[State.PZ],

                        state[State.VX], #Velocity X
                        state[State.VY],
                        state[State.VZ],

                        state[State.R1x], #Rotation matrix row 1 X
                        state[State.R1y],
                        state[State.R1z],

                        state[State.R2x], #Rotation matrix row 2 X
                        state[State.R2y],
                        state[State.R2z],

                        state[State.R3x], #Rotation matrix row 3 X
                        state[State.R3y],
                        state[State.R3z],

                        state[State.WX], #Angular velocity X
                        state[State.WY],
                        state[State.WZ],
                    ])

