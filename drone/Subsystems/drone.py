from pickletools import dis
from time import time
import numpy as np
import asyncio
from Subsystems.flight_controller import FlightController, State
from Subsystems.action import ActionSequence
from Subsystems.database import Database


class Drone():
    def __init__(self, flight_controller: FlightController | None = None, action_sequence : ActionSequence | None = None):
        self.state = State.generate()
        self.flight_controller = flight_controller
        self.time_stamp = 0
        self.update_frequency : float = FlightController.CONTROL_HZ
        self.action_sequence = action_sequence
        if self.flight_controller is None:
            raise ValueError("A flight controller must be provided.")
    def prompt_update_frequency(self):
        frequency = input("Enter the update frequency in seconds (real time is 100, as fast as possible is 0, 50 percent is 50): ")
        if (float(frequency) < 0):
            print("Negative??? Using default update frequency.")
        else:
            self.update_frequency = float(frequency)
    def get_database(self) -> Database:
        return self.flight_controller.database
    def step(self, action: np.ndarray, display: bool = False) -> None:
        self.state,self.time_stamp = self.flight_controller.step(action=action)
        if display:
            print(self.get_database().to_string())
    async def update(self, action : np.ndarray | None = None, display: bool = False) -> bool:
        wait_time = 0
        if (self.update_frequency > 0):
            wait_time = 1 / self.update_frequency
        await asyncio.sleep(wait_time)
        if action is None and self.action_sequence is not None:
            action = self.action_sequence.get_action(self.time_stamp)
        if (action is None):
            return False
        self.step(action, display=display)
        return True