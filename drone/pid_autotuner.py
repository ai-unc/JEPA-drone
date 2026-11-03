"""Config-based rate tuning for the standard ProjectAirSim X quad.

Calculated gains target a slow, damped rate response at nominal air density.
The model assumes SimulationConfigs.write()'s box frame and point-mass motors.
Rate control brakes rotation; it does not level the aircraft or hold altitude.
"""
from dataclasses import dataclass
import math
import json5
import numpy as np
from drone.Subsystems.PID import PID
from drone.Subsystems.data.data import Action


@dataclass(frozen=True)
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = 2.0
    integral_limit: float = 1.0

    def create(self):
        return PID(**vars(self))


class RatePIDAutotuner:
    # Existing gamepad_flight.py settings, preserved verbatim.
    DEFAULT_CONFIGS = {
        "roll": PIDConfig(-0.13),
        "pitch": PIDConfig(-0.2),
        "yaw": PIDConfig(0.8),
    }

    @classmethod
    def default_configs(cls):
        return dict(cls.DEFAULT_CONFIGS)

    @staticmethod
    def calculated_configs(configs, dt=0.01, response_time=0.25):
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
                for axis, gain in zip(("roll", "pitch", "yaw"), gains)}

    @classmethod
    def generate_configs(cls, configs=None, mode="calculated", **kwargs):
        if mode == "default":
            return cls.default_configs()
        if mode != "calculated":
            raise ValueError("mode must be 'default' or 'calculated'")
        if configs is None:
            raise ValueError("Calculated mode requires SimulationConfigs")
        return cls.calculated_configs(configs, **kwargs)

    @classmethod
    def generate_controllers(cls, configs=None, mode="calculated", **kwargs):
        generated = cls.generate_configs(configs, mode, **kwargs)
        return tuple(generated[axis].create() for axis in ("roll", "pitch", "yaw"))
