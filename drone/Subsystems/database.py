import numpy as np
from Subsystems.action import Action
from Subsystems.state import State

    
class Database:
    HISTORY_TIME: float = 0.5  # seconds
    SAVE_FREQUENCY: int = 20  # Hz
    SAVE_PERIOD: float = 1 / SAVE_FREQUENCY
    STEPS: int = int(HISTORY_TIME * SAVE_FREQUENCY)

    def __init__(self) -> None:
        self.step = 0
        self.action_history : list[np.ndarray] = []
        self.state_history : list[np.ndarray] = []
        self.time_history : list[np.double] = []

    def update_history(self, state: np.ndarray, action: np.ndarray, time_stamp: np.double, display : bool):
        #Make sure enough time has passed to update the state history
        if (self.step > 0 and self.time_history[self.step - 1] is not None and time_stamp - self.time_history[self.step - 1] < self.SAVE_PERIOD):
            return
        self.step = min(self.step, self.STEPS - 1)
        self.step = max(self.step, 0)

        self.state_history.append(state)
        self.action_history.append(action)
        self.time_history.append(time_stamp)
        
        if (self.step < self.STEPS - 1):
            self.step += 1
        else:
            self.state_history.pop(0)
            self.action_history.pop(0)
            self.time_history.pop(0)
        
        if display:
            print("----- Recent Step -----")
            print(self.step_to_string())
    def step_to_string(self,i = None):
        if (self.step == 0):
            return "No history available"
        if i is None:
            i = self.step - 1
        return (
            f"Step: {i+1}\n" +
            f"Action: {Action.to_string(self.action_history[i])}\n" +
            f"State: {State.to_string(self.state_history[i])}\n" +
            f"Time Step: {self.time_history[i]}"
        )
    def to_string(self) -> str:
        output = ""
        for i in range(self.step):
            output += (
                self.step_to_string(i) + "\n"
            )
        return output