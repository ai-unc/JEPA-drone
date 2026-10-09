import numpy as np
from Subsystems.action import Action, State

class Database:
    HISTORY_TIME: float = 1  # seconds
    SAVE_FREQUENCY: int = 100  # Hz
    STEPS: int = int(HISTORY_TIME * SAVE_FREQUENCY)

    def __init__(self) -> None:
        self.step = 0   # Number of stored entries (0-100)
        self.index = 0  # Next buffer position to write

        self.action_history: np.ndarray | None = None
        self.state_history: np.ndarray | None = None
        self.time_history = np.empty(self.STEPS, dtype=np.float64)

    def update_history(
        self,
        state: np.ndarray,
        action: np.ndarray,
        time_stamp: np.double,
        display: bool = False
    ) -> None:

        # Initialize buffers on first update
        if self.state_history is None:
            self.state_history = np.empty(
                (self.STEPS, *state.shape),
                dtype=state.dtype
            )
            self.action_history = np.empty(
                (self.STEPS, *action.shape),
                dtype=action.dtype
            )

        # Save data at current position
        self.state_history[self.index] = state
        self.action_history[self.index] = action
        self.time_history[self.index] = time_stamp

        # Advance circular buffer
        self.index = (self.index + 1) % self.STEPS
        self.step = min(self.step + 1, self.STEPS)

        if display:
            print("----- Recent Step -----")
            print(self.step_to_string())

    def get_history(self):
        """
        Returns histories in chronological order:
        oldest -> newest.
        """
        if self.step == 0:
            return None, None, None

        indices = (
            np.arange(self.step) + self.index - self.step
        ) % self.STEPS

        return (
            self.state_history[indices],
            self.action_history[indices],
            self.time_history[indices]
        )

    def step_to_string(self, i=None) -> str:
        if self.step == 0:
            return "No history available"

        if i is None:
            i = self.step - 1

        if not 0 <= i < self.step:
            raise IndexError("History step out of range")

        # Convert chronological index to buffer index
        buffer_index = (self.index - self.step + i) % self.STEPS

        return (
            f"Step: {i + 1}\n"
            f"Action: {Action.to_string(self.action_history[buffer_index])}\n"
            f"State: {State.to_string(self.state_history[buffer_index])}\n"
            f"Time Step: {self.time_history[buffer_index]}"
        )

    def to_string(self) -> str:
        return "\n".join(
            self.step_to_string(i)
            for i in range(self.step)
        )