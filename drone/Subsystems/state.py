import numpy as np

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