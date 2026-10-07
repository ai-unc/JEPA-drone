import numpy as np

class State():
    def __init__(
            self,
            position: np.ndarray = np.zeros(3,dtype=float),
            velocity: np.ndarray = np.zeros(3,dtype=float),
            rotation: np.ndarray = np.zeros((3,3),dtype=float),
            angular_velocity: np.ndarray = np.zeros(3,dtype=float),
            ):
        self.position = position
        self.velocity = velocity
        self.rotation = rotation
        self.angular_velocity = angular_velocity
    def to_string(self):
        return (
            f"position=({self.position[0]:+7.3f}, {self.position[1]:+7.3f}, {self.position[2]:+7.3f}) m \n" +
            f"velocity=({self.velocity[0]:+7.3f}, {self.velocity[1]:+7.3f}, {self.velocity[2]:+7.3f}) m/s \n " +
            f"rotation matrix=([\n" +
            f"  {self.rotation[0,0]:+7.3f}, {self.rotation[0,1]:+7.3f}, {self.rotation[0,2]:+7.3f}\n" +
            f"  {self.rotation[1,0]:+7.3f}, {self.rotation[1,1]:+7.3f}, {self.rotation[1,2]:+7.3f}\n" +
            f"  {self.rotation[2,0]:+7.3f}, {self.rotation[2,1]:+7.3f}, {self.rotation[2,2]:+7.3f}\n" +
            f"]) rad \n" +
            f"angular_velocity=({self.angular_velocity[0]:+7.3f}, {self.angular_velocity[1]:+7.3f}, {self.angular_velocity[2]:+7.3f}) rad/s"
        )