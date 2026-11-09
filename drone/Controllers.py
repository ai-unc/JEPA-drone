"""Config-based rate tuning for the standard ProjectAirSim X quad.

Calculated gains target a slow, damped rate response at the local air density.
The model assumes SimulationConfigs.write()'s box frame and point-mass motors.
Rate control brakes rotation; it does not level the aircraft or hold altitude.
"""
from dataclasses import dataclass, replace
import math
import json5
import numpy as np
from scipy.spatial.transform import Rotation
from drone.Subsystems.PID import PID
from drone.Subsystems.data.data import Action,State
from drone.simulations.physics_model import atmosphere, ROTOR_BASE_DENSITY, SEA_LEVEL_DENSITY

from enum import Enum

def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

gravity = 9.80665  # m/s^2. Positive because up is negative


@dataclass(frozen=True)
class PIDConfig:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    output_limit: float = 2.0
    integral_limit: float = 1.0

    def create(self):
        return PID(**vars(self))


@dataclass(frozen=True)
class RateThrustConfig:
    """One physics-derived setup for rate feedback and motor allocation.

    Motor thrust is in newtons. Collective values are per-motor action units,
    normalized by Action.ASSUMED_MAX_THRUST when sent to the simulator.
    """
    pid_configs: dict[str, PIDConfig]
    max_motor_thrust: float
    hover_collective: float
    recovery_collective: float
    response_time: float
    air_density: float
    gravity_magnitude: float
    altitude_m: float
    gamepad_thrust_to_weight: float

    # Optional multipliers for gamepad inputs AFTER deadzone processing and
    # BEFORE conversion to collective/rad/s. Never applied by RateController.
    throttle_sensitivity: float = 1.0
    roll_rate_sensitivity: float = 1.0
    pitch_rate_sensitivity: float = 1.0
    yaw_rate_sensitivity: float = 1.0

    def at_altitude(self, altitude_m):
        """Refresh environmental scaling without resetting any PID history."""
        density, local_gravity = atmosphere(altitude_m)
        ratio = density / self.air_density
        thrust = self.max_motor_thrust * ratio
        hover = self.hover_collective / ratio * local_gravity / self.gravity_magnitude
        sensitivity = min(1.0, self.gamepad_thrust_to_weight * hover / Action.ASSUMED_MAX_THRUST)
        return replace(
            self, pid_configs={axis: replace(pid, kp=pid.kp / ratio,
                                             ki=pid.ki / ratio, kd=pid.kd / ratio)
                               for axis, pid in self.pid_configs.items()},
            max_motor_thrust=thrust, hover_collective=hover,
            recovery_collective=min(Action.ASSUMED_MAX_THRUST / 2, .5 * hover),
            air_density=density, gravity_magnitude=local_gravity, altitude_m=altitude_m,
            throttle_sensitivity=sensitivity,
            roll_rate_sensitivity=math.sqrt(sensitivity),
            pitch_rate_sensitivity=math.sqrt(sensitivity),
            yaw_rate_sensitivity=math.sqrt(sensitivity),
        )


class RateController:
    """
    1. Can convert gamepad inputs (throttle, roll_vel, pitch_vel, yaw_vel) to collective thrust and target angular velocities.
    2. Can calculate motor outputs from collective thrust, target angular velocities, and current angular velocities.
    3. Tunes rate gains and recovery thrust together from SimulationConfigs.
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
        Gains.ROLL.value: PIDConfig(-0.05),
        Gains.PITCH.value: PIDConfig(-0.12),
        Gains.YAW.value: PIDConfig(0.8),
    }

    # Fixed nominal 1 kg frame and 4.18 N motors for baseline comparisons.
    # These values deliberately do not depend on the active simulation config.
    DEFAULT_CONTROL_CONFIG = RateThrustConfig(
        pid_configs=DEFAULT_CONFIGS,
        max_motor_thrust=Action.ASSUMED_MAX_THRUST,
        hover_collective=gravity / 4,
        recovery_collective=gravity / 8,
        response_time=0.25,
        air_density=atmosphere(0.0)[0],
        gravity_magnitude=gravity,
        altitude_m=0.0,
        gamepad_thrust_to_weight=2.0,
    )

    MAX_ROLL_RATE = 2.0
    MAX_PITCH_RATE = 2.0
    MAX_YAW_RATE = 2.0

    def __init__(self, mode: str = Modes.DEFAULT):
        self.roll_pid = self.DEFAULT_CONFIGS[self.Gains.ROLL.value].create()
        self.pitch_pid = self.DEFAULT_CONFIGS[self.Gains.PITCH.value].create()
        self.yaw_pid = self.DEFAULT_CONFIGS[self.Gains.YAW.value].create()
        self.mode = mode
        self.control_config = replace(self.DEFAULT_CONTROL_CONFIG, pid_configs=dict(self.DEFAULT_CONFIGS))
    def create(self, configs=None, dt=0.01, response_time=0.25, *, altitude_m=0.0):
        if (self.mode == RateController.Modes.CALCULATED):
            if configs is None:
                raise ValueError("Calculated mode requires simulation configs")
            self.control_config = self.calculate_control_config(configs, dt, response_time, altitude_m=altitude_m)
            pid_configs = self.control_config.pid_configs
        else:
            self.control_config = self.DEFAULT_CONTROL_CONFIG.at_altitude(altitude_m)
            # Keep the baseline PID gains fixed at every altitude.
            self.control_config = replace(self.control_config, pid_configs=dict(self.DEFAULT_CONFIGS))
            pid_configs = self.control_config.pid_configs
        self.roll_pid = pid_configs[self.Gains.ROLL.value].create()
        self.pitch_pid = pid_configs[self.Gains.PITCH.value].create()
        self.yaw_pid = pid_configs[self.Gains.YAW.value].create()
    def update_altitude(self, altitude_m):
        self.control_config = self.control_config.at_altitude(altitude_m)
        if self.mode == self.Modes.CALCULATED:
            for axis, pid in (("roll", self.roll_pid), ("pitch", self.pitch_pid), ("yaw", self.yaw_pid)):
                gain = self.control_config.pid_configs[axis]
                pid.kp, pid.ki, pid.kd = gain.kp, gain.ki, gain.kd

    @property
    def recovery_collective(self):
        return self.control_config.recovery_collective

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
               return response



    @staticmethod
    def calculate_control_config(configs, dt=0.01, response_time=0.25,
                                 *, gamepad_thrust_to_weight=2.0, altitude_m=0.0) -> RateThrustConfig:
        """Tune rate gains and recovery collective from one shared physics model.

        Larger response_time gives softer P-only rate control. Recovery thrust
        is capped at half nominal hover to brake rotation at closed throttle.
        Motor commands are normalized
        by Action.ASSUMED_MAX_THRUST, so gains account for actual motor power.
        Recalculate whenever physics configs change, after configs.write().

        Optional gamepad multipliers are returned as metadata only. Throttle
        scaling caps full stick at gamepad_thrust_to_weight times weight
        (local density); hover is 1/gamepad_thrust_to_weight stick when enough
        thrust is available. This intentionally limits accessible top power.
        Rate multipliers use sqrt(throttle_sensitivity), a softer-stick comfort
        heuristic, not compensation for plant gain. Closed-loop rate gains
        already account for motor authority. Apply multipliers externally to
        normalized gamepad inputs after deadzones, before unit conversion.
        Do not apply rate multipliers to attitude-controller output rates.
        """
        positive = [configs.mass, configs.arm_length, configs.thrust_coef,
                    configs.torque_coef, configs.MAX_RPM, dt, response_time,
                    gamepad_thrust_to_weight]
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Mass, arm length, coefficients, RPM, timing and gamepad thrust/weight must be positive and finite")
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
        # FastPhysicsBody::CalculateInertia aggregates diagonal terms about
        # the root origin, not the combined center of mass.
        inertia = frame_mass / 12 * (np.sum(dimensions**2) - dimensions**2)
        inertia += motor_mass * (np.sum(positions**2) - np.sum(positions**2, axis=0))
        # ProjectAirSim RotorSettings::CalcMaxThrustAndTorque formulas.
        density, local_gravity = atmosphere(altitude_m)
        # RotorSettings uses 1.225 base density; Robot::UpdateActuators then
        # applies density / EarthUtils::kSeaLevelAirDensity to force AND torque.
        rpm_factor = ROTOR_BASE_DENSITY * (configs.MAX_RPM / 60)**2 * density / SEA_LEVEL_DENSITY
        thrust = configs.thrust_coef * rpm_factor * diameter**4
        torque = configs.torque_coef * rpm_factor * diameter**5 / (2 * math.pi)
        # Mixer corrections affect all four motors; upward thrust is -Z.
        authority = np.array([-4 * configs.arm_length / math.sqrt(2) * thrust,
                                -4 * configs.arm_length / math.sqrt(2) * thrust,
                                4 * torque]) / Action.ASSUMED_MAX_THRUST / inertia
        response = max(response_time, 12 * configs.motor_time_constant, 20 * dt)
        gains = 1 / (authority * response)
        pid_configs = {axis: PIDConfig(float(gain), output_limit=Action.ASSUMED_MAX_THRUST / 4)
                for axis, gain in zip((RateController.Gains.ROLL.value, RateController.Gains.PITCH.value, RateController.Gains.YAW.value), gains)}
        # ProjectAirSim FastPhysicsBody initializes translational mass from
        # the root link ONLY (SetMass(root_link.GetInertial().GetMass())).
        # SimulationConfigs.mass includes four motors, but their mass is not
        # accumulated by that backend. Match its root mass for thrust scaling;
        # keep the full link geometry above for rotational inertia.
        flight_mass = frame_mass
        hover_collective = flight_mass * local_gravity / (4 * thrust) * Action.ASSUMED_MAX_THRUST
        recovery_collective = min(Action.ASSUMED_MAX_THRUST / 2, .5 * hover_collective)
        thrust_to_weight = 4 * thrust / (flight_mass * local_gravity)
        throttle_sensitivity = min(1.0, gamepad_thrust_to_weight / thrust_to_weight)
        rate_sensitivity = math.sqrt(throttle_sensitivity)
        return RateThrustConfig(
            pid_configs=pid_configs,
            max_motor_thrust=thrust,
            hover_collective=hover_collective,
            recovery_collective=recovery_collective,
            response_time=response,
            air_density=density,
            gravity_magnitude=local_gravity,
            altitude_m=altitude_m,
            gamepad_thrust_to_weight=gamepad_thrust_to_weight,
            throttle_sensitivity=throttle_sensitivity,
            roll_rate_sensitivity=rate_sensitivity,
            pitch_rate_sensitivity=rate_sensitivity,
            yaw_rate_sensitivity=rate_sensitivity,
        )


    @staticmethod
    def allocate_motor_outputs(collective, roll, pitch, yaw, limit, recovery):
        """Prioritize roll/pitch, then fit yaw into remaining motor headroom.

        Raise collective only as needed, and only up to max(requested,
        recovery). Recovery is half nominal hover, so even saturated braking
        at closed throttle cannot support the vehicle's weight.
        """
        requested = float(np.clip(collective, 0.0, limit))
        budget = max(requested, float(np.clip(recovery, 0.0, limit / 2)))
        rp = np.array([-roll - pitch, roll - pitch,
                       -roll + pitch, roll + pitch])
        peak = float(np.max(np.abs(rp)))
        if peak > 0:
            rp *= min(1.0, budget / peak, limit / (2 * peak))
        # Preserve roll/pitch by reducing collective near full throttle.
        yaw_vector = np.array([-yaw, yaw, yaw, -yaw])
        factor = 1.0
        # Feasible collective interval: -min(correction) <= base <=
        # limit-max(correction), additionally base <= budget. Solve these
        # linear inequalities for the largest attainable yaw fraction.
        for i in range(4):
            if yaw_vector[i] < 0:
                factor = min(factor, (budget + rp[i]) / -yaw_vector[i])
            for j in range(4):
                delta = yaw_vector[i] - yaw_vector[j]
                if delta > 0:
                    factor = min(factor, (limit - rp[i] + rp[j]) / delta)
        corrections = rp + max(0.0, factor) * yaw_vector
        base = float(np.clip(requested, -np.min(corrections),
                             min(budget, limit - np.max(corrections))))
        return np.clip(base + corrections, 0.0, limit)

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

        fl, fr, bl, br = self.allocate_motor_outputs(
            collective_thrust, roll_output, pitch_output, yaw_output,
            max_thrust, self.recovery_collective,
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
        Gains.YAW.value: PIDConfig(0.5,output_limit=8.0),
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
    ) -> tuple[tuple[np.double, np.double, np.double]]:
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


class VelocityController:
    """
    This class controls the velocity of the drone.
    Assuming attitude control works, there will be no automatic tuning for velocity.
    """
    MAX_X_VELOCITY = 8 #m/s
    MAX_Y_VELOCITY = 8 #m/s
    MAX_Z_VELOCITY = 3 #m/s -> Vertical

    MAX_ACCELERATION = 3 #m/s^2

    class Gains(Enum):
        X = "x_velocity"
        Y = "y_velocity"
        Z = "z_velocity"
    CONFIGS = {
        Gains.X.value: PIDConfig(0.3,output_limit=MAX_ACCELERATION),
        Gains.Y.value: PIDConfig(0.3,output_limit=MAX_ACCELERATION),
        # PI trims small mass/thrust/gravity biases; P alone leaves a steady
        # vertical speed offset. Integral trim is bounded to +/-1 m/s^2.
        Gains.Z.value: PIDConfig(0.8, ki=0.05, output_limit=MAX_ACCELERATION,
                                 integral_limit=20.0),
    }

    def __init__(self):
        self.x_pid = self.CONFIGS[self.Gains.X.value].create()
        self.y_pid = self.CONFIGS[self.Gains.Y.value].create()
        self.z_pid = self.CONFIGS[self.Gains.Z.value].create()
    def reset(self):
        self.x_pid.reset()
        self.y_pid.reset()
        self.z_pid.reset()
    @staticmethod
    def relative_to_world(forward, horizontal, yaw) -> tuple[np.double, np.double]:
        """Convert heading-relative forward/right velocity to world X and Y
        """
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (np.double(cosine * forward - sine * horizontal),
                np.double(sine * forward + cosine * horizontal))
    def calculate_attitudes(
            self,
            target_velocity: np.ndarray,
            current_velocity: np.ndarray,
            dt: np.double,
            yaw_angle: np.double = 0.0,
            *,
            hover_collective: float,
            gravity_magnitude: float = gravity,
    ) -> tuple[np.ndarray, float]:
        """Return (target rotation matrix, per-motor collective action).

        Velocities are world-frame m/s with positive Z down. yaw_angle
        is the desired world heading in radians. Pass hover_collective
        from RateController.control_config; it supplies the vehicle's
        mass/thrust scaling without treating action units as newtons.

        The thrust magnitude includes gravity, vertical acceleration,
        and the additional thrust needed for horizontal acceleration.
        """
        target = np.asarray(target_velocity, dtype=float)
        current = np.asarray(current_velocity, dtype=float)

        # Preserve the previous trim so saturation can reject windup below.
        previous_z_integral = self.z_pid.integral
        acceleration = np.array([
            self.x_pid.update(target[0], current[0], dt),
            self.y_pid.update(target[1], current[1], dt),
            self.z_pid.update(target[2], current[2], dt),
        ])
        # a_world = gravity*e_z - (total_thrust/mass)*R[:, 2].
        # Rotate the requested horizontal acceleration into heading axes
        # to solve roll/pitch while preserving the specified Euler yaw.
        cosine, sine = math.cos(yaw_angle), math.sin(yaw_angle)
        ax = cosine * acceleration[0] + sine * acceleration[1]
        ay = -sine * acceleration[0] + cosine * acceleration[1]
        upward_acceleration = gravity_magnitude - acceleration[2]
        # With the configured +/-3 m/s^2 limits this is always positive.
        if upward_acceleration <= 0:
            upward_acceleration = 1e-6
        pitch_angle = math.atan2(-ax, upward_acceleration)
        roll_angle = math.atan2(ay, math.hypot(ax, upward_acceleration))
        rotation_matrix = Rotation.from_euler(
            "xyz", [roll_angle, pitch_angle, yaw_angle], degrees=False
        ).as_matrix()
        thrust_per_mass = math.sqrt(ax**2 + ay**2 + upward_acceleration**2)
        requested_collective = hover_collective * thrust_per_mass / gravity_magnitude
        collective = float(np.clip(requested_collective, 0.0, Action.ASSUMED_MAX_THRUST))
        z_error = target[2] - current[2]
        acceleration_saturated = (
            self.z_pid.output_limit is not None
            and abs(acceleration[2]) >= self.z_pid.output_limit
            and z_error * acceleration[2] > 0
        )
        # Negative Z error asks for more upward thrust. Do not build up trim
        # when that request exceeds available thrust; allow unwinding instead.
        thrust_saturated = requested_collective >= Action.ASSUMED_MAX_THRUST and z_error < 0
        if acceleration_saturated or thrust_saturated:
            self.z_pid.integral = previous_z_integral
        return rotation_matrix, collective
class PositionController:
    class Gains(Enum):
        HORIZONTAL = "horizontal"
        FORWARD = "forward"
        VERTICAL = "vertical"
    MAX_FORWARD_VELOCITY = 7 #m/s
    MAX_HORIZONTAL_VELOCITY = 9
    MAX_VERTICAL_VELOCITY = 5
    CONFIGS = {
            Gains.HORIZONTAL.value: PIDConfig(0.8,output_limit=MAX_HORIZONTAL_VELOCITY),
            Gains.FORWARD.value: PIDConfig(0.2,output_limit=MAX_FORWARD_VELOCITY),
            Gains.VERTICAL.value: PIDConfig(0.7,output_limit=MAX_VERTICAL_VELOCITY),
        }
    def __init__(self):
        self.horizontal_pid = self.CONFIGS[self.Gains.HORIZONTAL.value].create()
        self.forward_pid = self.CONFIGS[self.Gains.FORWARD.value].create()
        self.vertical_pid = self.CONFIGS[self.Gains.VERTICAL.value].create()
    def reset(self):
        self.horizontal_pid.reset()
        self.forward_pid.reset()
        self.vertical_pid.reset()
    def calculate_velocities(
            self,
            target_position : np.ndarray,
            current_position : np.ndarray,
            current_yaw : np.double,
            dt = 0.01
    ) -> tuple[np.double,np.double,np.double]:
        """Return heading-relative (right, forward, down) velocities in m/s.

        Positions are world-frame meters; current_yaw is in radians. Only
        yaw is used so vertical control stays aligned with world Z.
        """
        x0, y0, z0 = current_position
        x1, y1, z1 = target_position
        x_error, y_error, z_error = x1 - x0, y1 - y0, z1 - z0
        cosine, sine = math.cos(current_yaw), math.sin(current_yaw)
        forward_error = cosine * x_error + sine * y_error
        horizontal_error = -sine * x_error + cosine * y_error

        horizontal_velocity = self.horizontal_pid.update(horizontal_error, 0.0, dt)
        forward_velocity = self.forward_pid.update(forward_error, 0.0, dt)
        vertical_velocity = self.vertical_pid.update(z_error, 0.0, dt)
        return horizontal_velocity, forward_velocity, vertical_velocity


class Controllers():
    RATE = "rate"
    ATTITUDE = "attitude"
    VELOCITY = "velocity"
    WAYPOINT = "waypoint"
    @staticmethod
    def prompt_mode():
        """
        Prompt the user to get what controller they want, then return the corresponding controller.
        Does not include position since how are you going to fly that... waypoint is for autonomous flight
        """
        while True:
            response = input("Select Controller (" + Controllers.RATE + "/" + Controllers.ATTITUDE + "/" + Controllers.VELOCITY + "): ").strip().lower() or Controllers.RATE
            if response in (Controllers.RATE, Controllers.ATTITUDE, Controllers.VELOCITY):
                break
        return response
    @staticmethod
    def calculate_everything(
            position_controller : PositionController,
            velocity_controller : VelocityController,
            attitude_controller : AttitudeController,
            rate_controller : RateController,
            target_position : np.ndarray,
            target_yaw : np.double,
            current_state : np.ndarray,
            dt: float = 0.01,
    ) -> np.ndarray:
        """Calculate motor commands for a world-frame position and yaw heading.

        Position uses metres with positive Z down; target_yaw is in radians.
        dt is the control interval in seconds. Keep the rate controller's
        altitude-dependent configuration updated at the call site.
        Reuse the controllers between calls to preserve their PID history.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Control interval must be positive and finite")
        config = rate_controller.control_config

        current_position = State.get_position(current_state)
        current_rotation = State.get_rotation(current_state)
        current_yaw = np.arctan2(current_rotation[1, 0], current_rotation[0, 0])
        horizontal_velocity, forward_velocity, vertical_velocity = position_controller.calculate_velocities(
            target_position, current_position, current_yaw=current_yaw, dt=dt,
        )
        # The velocity controller consumes world-frame velocities.
        world_x, world_y = VelocityController.relative_to_world(
            forward_velocity, horizontal_velocity, current_yaw,
        )
        target_velocity = np.array([world_x, world_y, vertical_velocity], dtype=float)
        target_rotation, collective_thrust = velocity_controller.calculate_attitudes(
            target_velocity, State.get_velocity(current_state), dt=dt,
            yaw_angle=target_yaw,
            hover_collective=config.hover_collective,
            gravity_magnitude=config.gravity_magnitude,
        )
        roll_rate, pitch_rate, yaw_rate = attitude_controller.calculate_rates(
            target_rotation, current_rotation, dt=dt,
        )
        return rate_controller.calculate_action(
            collective_thrust=collective_thrust,
            target_roll_rate=roll_rate,
            target_pitch_rate=pitch_rate,
            target_yaw_rate=yaw_rate,
            angular_velocity=State.get_angular_velocity(current_state),
            dt=dt,
        )
