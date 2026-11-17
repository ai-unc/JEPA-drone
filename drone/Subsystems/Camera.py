import numpy as np

"""
- Images are grayscale to mimick Oak D lite
Turn 640,480 pixel image to 64,64 matrix containing depth and obstacle class
Depth is a float. Obstacle class is a str such as "obstacle"
"""
class Camera:
    FREQUENCY = 100 #Hz
    IMAGE_RESOLUTION = 640,480
    MAX_DEPTH = 3 #Meters
    MIN_DEPTH = 0.5 #Meters

    DEPTH_RESOLUTION = 64,64
    DEPTH_VALUES = 0,"obstacle"

    @staticmethod
    def encode_image(image):
        """
        a "cheat" method to convert what the drone sees to labeled depths
        """
        return None

    def __init__(self):
        self.image = None #640 * 480 matrix
        self.obstacle = None #NOT of raw images, but of labeled vision (64 x 64)