import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

#Used for gamepad flight to make flight tolerable
class PID:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False

    def update(self, target, current, dt):
        error = target - current

        # Integral
        self.integral += error * dt

        if self.integral_limit is not None:
            self.integral = clamp(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            )

        # Derivative
        if self.initialized and dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
            self.initialized = True

        self.previous_error = error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        if self.output_limit is not None:
            output = clamp(
                output,
                -self.output_limit,
                self.output_limit
            )

        return output

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.initialized = False