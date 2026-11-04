"""Config-based rate tuning for the standard ProjectAirSim X quad.

Calculated gains target a slow, damped rate response at nominal air density.
The model assumes SimulationConfigs.write()'s box frame and point-mass motors.
Rate control brakes rotation; it does not level the aircraft or hold altitude.
"""
from dataclasses import dataclass
import math
import json5
import numpy as np
from scipy.spatial.transform import Rotation
from drone.Subsystems.PID import PID
from drone.Subsystems.data.data import Action

from enum import Enum

def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

class Controllers():
    RATE = "rate"
    ATTITUDE = "attitude"
    VELOCITY = "velocity"
    @staticmethod
    def prompt_mode():
        """
        Prompt the user to get what controller they want, then return the corresponding controller.
        """
        while True:
            response = input("Select Controller (" + Controllers.RATE + "/" + Controllers.ATTITUDE + "/" + Controllers.VELOCITY + "): ").strip().lower() or Controllers.RATE
            if response in (Controllers.RATE, Controllers.ATTITUDE, Controllers.VELOCITY):
                break
        return response

@dataclass(frozen=True)
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = 2.0
    integral_limit: float = 1.0

    def create(self):
        return PID(**vars(self))


class RateController:
    """
    1. Can convert gamepad inputs (throttle, roll_vel, pitch_vel, yaw_vel) to collective thrust and target angular velocities.
    2. Can calculate motor outputs from collective thrust, target angular velocities, and current angular velocities.
    3. Can calculate PID gains from SimulationConfigs.
    --Ethan
    """
    class Modes:
        DEFAULT = "default"
        CALCULATED = "calculated"
    class Gains(Enum):
        ROLL = "roll"
        PITCH = "pitch"
        YAW = "yaw"

    DEFAULT_CONFIGS = {
        Gains.ROLL.value: PIDConfig(-0.13),
        Gains.PITCH.value: PIDConfig(-0.2),
        Gains.YAW.value: PIDConfig(0.8),
    }

    MAX_ROLL_RATE = 2.0
    MAX_PITCH_RATE = 2.0
    MAX_YAW_RATE = 2.0
    
    def __init__(self, mode : str):
        self.roll_pid = self.DEFAULT_CONFIGS[self.Gains.ROLL.value].create()
        self.pitch_pid = self.DEFAULT_CONFIGS[self.Gains.PITCH.value].create()
        self.yaw_pid = self.DEFAULT_CONFIGS[self.Gains.YAW.value].create()
        self.mode = mode
    def create(self, configs = None, dt=0.01, response_time=0.25):
        if self.mode == self.Modes.CALCULATED and configs is not None:
            pid_configs = self.calculate_rate_gains(configs, dt, response_time)
        else:
            pid_configs = self.DEFAULT_CONFIGS
        self.roll_pid = pid_configs[self.Gains.ROLL.value].create()
        self.pitch_pid = pid_configs[self.Gains.PITCH.value].create()
        self.yaw_pid = pid_configs[self.Gains.YAW.value].create()
    def print_configs(self):
        print("Rate PID settings:")
        print("Roll:", self.roll_pid.to_string())
        print("Pitch:", self.pitch_pid.to_string())
        print("Yaw:", self.yaw_pid.to_string())
    @staticmethod
    def prompt_mode():
        """
        Prompt the user to get what mode they want, then return the corresponding gains.
        """
        while True:
           response = input("Select Rate Gain Settings (" + RateController.Modes.DEFAULT + "/" + RateController.Modes.CALCULATED + "): ").strip().lower() or RateController.Modes.CALCULATED
           if response in (RateController.Modes.DEFAULT, RateController.Modes.CALCULATED):
               break



    @staticmethod
    def calculate_rate_gains(configs, dt=0.01, response_time=0.25):
        """Generate conservative P-only gains; larger response_time is softer.

        No integral windup or derivative kick. Motor commands are normalized
        by Action.ASSUMED_MAX_THRUST, so gains account for actual motor power.
        Recalculate whenever physics configs change, after configs.write().
        """
        positive = [configs.mass, configs.arm_length, configs.thrust_coef,
                    configs.torque_coef, configs.MAX_RPM, dt, response_time]
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Mass, arm length, coefficients, RPM and timing must be positive and finite")
        if not math.isfinite(configs.motor_time_constant) or configs.motor_time_constant < 0:
            raise ValueError("Motor time constant must be finite and nonnegative")
        with open(configs.file, encoding="utf-8") as stream:
            robot = json5.load(stream)
        frame = next(link for link in robot["links"] if link["name"] == "Frame")
        box = frame["inertial"]["inertia"]["geometry"]["box"]["size"]
        dimensions = np.array([float(v) for v in box.split()])
        if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions) & (dimensions > 0)):
            raise ValueError("Expected three positive frame box dimensions")
        motors = [link for link in robot["links"] if link["name"].startswith("Prop_")]
        if len(motors) != 4 or any(m["inertial"]["inertia"]["type"] != "point-mass" for m in motors):
            raise ValueError("Expected four point-mass motors")
        rotors = [a for a in robot["actuators"] if a["type"] == "rotor"]
        diameters = [r["rotor-settings"]["propeller-diameter"] for r in rotors]
        if len(rotors) != 4 or not all(math.isfinite(d) and d > 0 and d == diameters[0] for d in diameters):
            raise ValueError("Expected four identical positive propeller diameters")
        diameter = diameters[0]
        scale = configs.mass / configs.DEFAULT_TOTAL_MASS
        frame_mass = configs.DEFAULT_FRAME_MASS * scale
        motor_mass = configs.DEFAULT_MOTOR_MASS * scale
        positions = np.array(configs.calculate_motor_positions(configs.arm_length))
        center = motor_mass * positions.sum(axis=0) / configs.mass
        inertia = frame_mass / 12 * (np.sum(dimensions**2) - dimensions**2)
        inertia += frame_mass * (np.dot(center, center) - center**2)
        offsets = positions - center
        inertia += motor_mass * (np.sum(offsets**2) - np.sum(offsets**2, axis=0))
        # ProjectAirSim RotorSettings::CalcMaxThrustAndTorque formulas.
        rpm_factor = 1.225 * (configs.MAX_RPM / 60)**2
        thrust = configs.thrust_coef * rpm_factor * diameter**4
        torque = configs.torque_coef * rpm_factor * diameter**5 / (2 * math.pi)
        # Mixer corrections affect all four motors; upward thrust is -Z.
        authority = np.array([-4 * configs.arm_length / math.sqrt(2) * thrust,
                                -4 * configs.arm_length / math.sqrt(2) * thrust,
                                4 * torque]) / Action.ASSUMED_MAX_THRUST / inertia
        response = max(response_time, 12 * configs.motor_time_constant, 20 * dt)
        gains = 1 / (authority * response)
        return {axis: PIDConfig(float(gain), output_limit=Action.ASSUMED_MAX_THRUST / 4)
                for axis, gain in zip((RateController.Gains.ROLL.value, RateController.Gains.PITCH.value, RateController.Gains.YAW.value), gains)}
    
    @staticmethod
    def normalize_motor_outputs(outputs, max_thrust):
        """
        Fits motor outputs into [0, max_thrust] while preserving
        the relative differences between motors as much as possible.
        """
        outputs = np.array(outputs, dtype=float)

        minimum = np.min(outputs)
        maximum = np.max(outputs)

        # If the requested differential range is larger than the
        # physical motor range, scale the differential commands down.
        span = maximum - minimum

        if span > max_thrust:
            center = (maximum + minimum) / 2.0

            outputs = center + (
                outputs - center
            ) * (max_thrust / span)

        # Recalculate after scaling
        minimum = np.min(outputs)
        maximum = np.max(outputs)

        # Shift all motors downward together if any motor exceeds max.
        if maximum > max_thrust:
            outputs -= maximum - max_thrust

        # Shift all motors upward together if any motor is below zero.
        minimum = np.min(outputs)

        if minimum < 0:
            outputs -= minimum

        return np.clip(outputs, 0.0, max_thrust)
        
    @staticmethod
    def gamepad_to_target_rates(throttle,target_roll_vel,target_pitch_vel,target_yaw_vel):
        """
        Converts gamepad inputs to target angular velocities.
        """

        # Controller input [-1, 1] -> target angular velocity
        target_roll_vel *= RateController.MAX_ROLL_RATE
        target_pitch_vel *= RateController.MAX_PITCH_RATE
        target_yaw_vel *= RateController.MAX_YAW_RATE
        collective_thrust = throttle * Action.ASSUMED_MAX_THRUST

        return collective_thrust, target_roll_vel, target_pitch_vel, target_yaw_vel

    
    def calculate_action(
        self,
        collective_thrust : np.double,
        target_roll_rate : np.double,
        target_pitch_rate : np.double,
        target_yaw_rate : np.double,
        angular_velocity: np.ndarray,
        dt: np.double
    ) -> np.ndarray:
        # Maximum commanded rotational velocities in rad/s

        max_thrust = Action.ASSUMED_MAX_THRUST

        current_roll_rate = angular_velocity[0]
        current_pitch_rate = angular_velocity[1]
        current_yaw_rate = angular_velocity[2]

        # Angular velocity PID controllers
        roll_output = self.roll_pid.update(
            target_roll_rate,
            current_roll_rate,
            dt
        )

        pitch_output = self.pitch_pid.update(
            target_pitch_rate,
            current_pitch_rate,
            dt
        )

        yaw_output = self.yaw_pid.update(
            target_yaw_rate,
            current_yaw_rate,
            dt
        )

        corrections = np.array([
            -roll_output - pitch_output - yaw_output,
            roll_output - pitch_output + yaw_output,
            -roll_output + pitch_output + yaw_output,
            roll_output + pitch_output - yaw_output
        ])

        required_headroom = max(0.0, np.max(corrections))


        base = min(
            collective_thrust,
            max_thrust - required_headroom
        )

        """ print(
            "target_roll_rate:", target_roll_rate,
            "current_roll_rate:", current_roll_rate,
            "target_pitch_rate:", target_pitch_rate,
            "current_pitch_rate:", current_pitch_rate,
            "target_yaw_rate:", target_yaw_rate,
            "current_yaw_rate:", current_yaw_rate
        )"""

        # X-quad mixer
        fl = base - roll_output - pitch_output - yaw_output
        fr = base + roll_output - pitch_output + yaw_output
        bl = base - roll_output + pitch_output + yaw_output
        br = base + roll_output + pitch_output - yaw_output

        # Desaturate while preserving motor differences
        fl, fr, bl, br = RateController.normalize_motor_outputs(
            [fl, fr, bl, br],
            max_thrust
        )

        return Action.generate(
            np.array([
                np.double(fl),
                np.double(fr),
                np.double(bl),
                np.double(br),
            ],
            dtype=np.double)
        )



class AttitudeController:
    """
    This class controls orientation of the drone. 
    Assuming rate control works, there will be no automatic tuning for atitude.
    """
    class Gains(Enum):
        ROLL = "roll"
        PITCH = "pitch"
        YAW = "yaw"
    CONFIGS = {
        Gains.ROLL.value: PIDConfig(3,output_limit=20.0),
        Gains.PITCH.value: PIDConfig(2,output_limit=12.0),
        Gains.YAW.value: PIDConfig(0.8,output_limit=2.0),
    }
    MAX_ROLL_ANGLE = 30 #degrees
    MAX_PITCH_ANGLE = 30 #degrees
    MAX_YAW_ANGLE = 180 #degrees
    def __init__(self):
        self.roll_pid = self.CONFIGS[self.Gains.ROLL.value].create()
        self.pitch_pid = self.CONFIGS[self.Gains.PITCH.value].create()
        self.yaw_pid = self.CONFIGS[self.Gains.YAW.value].create()
    def reset(self):
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
    def calculate_rates(
        self,
        target_rotation_matrix: np.ndarray,
        current_rotation_matrix: np.ndarray,
        dt: np.double
    ) -> tuple[np.double, np.double, np.double]:
        """
        Calculate body-frame target angular velocities in radians per second.

        Both matrices map body-frame vectors into the world frame.
        """
        relative_rotation = current_rotation_matrix.T @ target_rotation_matrix
        roll_error, pitch_error, yaw_error = Rotation.from_matrix(
            relative_rotation
        ).as_rotvec()

        roll_rate = self.roll_pid.update(
            roll_error,
            0.0,
            dt
        )

        pitch_rate = self.pitch_pid.update(
            pitch_error,
            0.0,
            dt
        )

        yaw_rate = self.yaw_pid.update(
            yaw_error,
            0.0,
            dt
        )

        return roll_rate, pitch_rate, yaw_rate
    @staticmethod
    def gamepad_to_attitude(
        roll_input, pitch_input, current_rotation_matrix: np.ndarray | None = None, yaw_input: np.double | None = None
    ) -> np.ndarray:
        """
        Converts normalized (-1 to 1) gamepad inputs to target attitude
        Preserve the current heading when a current rotation is supplied.
        Yaw stick commands are handled separately by rate control.
        For the sake of testing yaw attitude I have this able to use yaw stick commands.
        This outputs a 3x3 rotation matrix
        --Ethan
        """

        # Gamepad.attitude_control_inputs() already applies the pitch sign.
        target_roll = roll_input * AttitudeController.MAX_ROLL_ANGLE
        target_pitch = pitch_input * AttitudeController.MAX_PITCH_ANGLE
        if (yaw_input is not None):
            target_yaw = yaw_input * AttitudeController.MAX_YAW_ANGLE
            return Rotation.from_euler(
                "xyz", [target_roll, target_pitch, target_yaw], degrees=True
            ).as_matrix()
        tilt = Rotation.from_euler(
            "xyz", [target_roll, target_pitch, 0.0], degrees=True
        ).as_matrix()
        if current_rotation_matrix is None:
            return tilt
        # Heading of the body X axis in the world horizontal plane.
        yaw = np.arctan2(current_rotation_matrix[1, 0], current_rotation_matrix[0, 0])
        return Rotation.from_euler("z", yaw).as_matrix() @ tilt