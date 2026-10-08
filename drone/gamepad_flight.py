import inputs
import asyncio
from Subsystems.Gamepad import Gamepad

from Subsystems.drone import Drone
from Subsystems.action import Action, ActionSequence
import numpy as np
from drone.Subsystems.state import State
from simulations.simulation import ProjectAirSimSimulation
from Subsystems.PID import PID

#There is a bunch of AI Generated code here. This was used to fly around and collect data.
#To be flyable by a human there needs to be several PID controllers to control rotational velocity
#Which is why this file is so massive

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

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

def controller_to_action(
    throttle,
    target_roll_vel,
    target_pitch_vel,
    target_yaw_vel,
    angular_velocity: np.ndarray,
    dt: np.double,
    roll_pid: PID,
    pitch_pid: PID,
    yaw_pid: PID
) -> np.ndarray:
    # Maximum commanded rotational velocities in rad/s
    MAX_ROLL_RATE = 2.0
    MAX_PITCH_RATE = 2.0
    MAX_YAW_RATE = 2.0

    max_thrust = Action.ASSUMED_MAX_THRUST

    # Controller input [-1, 1] -> target angular velocity
    target_roll_vel *= MAX_ROLL_RATE
    target_pitch_vel *= MAX_PITCH_RATE
    target_yaw_vel *= MAX_YAW_RATE

    current_roll_rate = angular_velocity[0]
    current_pitch_rate = angular_velocity[1]
    current_yaw_rate = angular_velocity[2]

    # Angular velocity PID controllers
    roll_output = roll_pid.update(
        target_roll_vel,
        current_roll_rate,
        dt
    )

    pitch_output = pitch_pid.update(
        target_pitch_vel,
        current_pitch_rate,
        dt
    )

    yaw_output = yaw_pid.update(
        target_yaw_vel,
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

    requested_base = throttle * max_thrust

    base = min(
        requested_base,
        max_thrust - required_headroom
    )

    """ print(
        "target_roll_vel:", target_roll_vel,
        "current_roll_vel:", current_roll_rate,
        "target_pitch_vel:", target_pitch_vel,
        "current_pitch_vel:", current_pitch_rate,
        "target_yaw_vel:", target_yaw_vel,
        "current_yaw_vel:", current_yaw_rate
    )"""

    # X-quad mixer
    fl = base - roll_output - pitch_output - yaw_output
    fr = base + roll_output - pitch_output + yaw_output
    bl = base - roll_output + pitch_output + yaw_output
    br = base + roll_output + pitch_output - yaw_output

    # Desaturate while preserving motor differences
    fl, fr, bl, br = normalize_motor_outputs(
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

async def main():
    max_time = float(input("Enter the maximum simulation time: "))
    simulation = ProjectAirSimSimulation()
    simulation.log_prompt()
    drone = Drone(flight_controller=simulation)
    gamepad = Gamepad()

    action = None

    #Left Stick y is throttle
    #Left Stick x is roll
    #Right Stick y is pitch
    #Right Stick x is yaw
    try:
        simulation.start()
        simulation.set_pose_to_random_in_sky()

        roll_pid = PID(
            kp=-0.13,
            ki=0,
            kd=-0.000,
            output_limit=2,
            integral_limit=1.0
        )

        pitch_pid = PID(
            kp=-0.2,
            ki=0,
            kd=0.000,
            output_limit=2,
            integral_limit=1.0
        )

        yaw_pid = PID(
            kp=0.8,
            ki=0.00,
            kd=0.0,
            output_limit=2,
            integral_limit=1.0
        )
        
        drone.action_sequence = None

        while (drone.time_stamp < max_time):
            throttle = gamepad.get_joystick(Gamepad.Inputs.LEFT_Y)
            roll     = gamepad.get_joystick(Gamepad.Inputs.LEFT_X)
            pitch    = gamepad.get_joystick(Gamepad.Inputs.RIGHT_Y) * -1
            yaw      = gamepad.get_joystick(Gamepad.Inputs.RIGHT_X)
            THRESHOLD = 0.2


            # Apply deadzone threshold
            if abs(throttle) < THRESHOLD:
                throttle = 0.0
            if abs(roll) < THRESHOLD:
                roll = 0.0
            if abs(pitch) < THRESHOLD:
                pitch = 0.0
            if abs(yaw) < THRESHOLD:
                yaw = 0.0
            angular_velocity = State.get_angular_velocity(drone.state)
            action = controller_to_action(
                throttle,
                roll,
                pitch,
                yaw,
                angular_velocity,
                drone.flight_controller.CONTROL_PERIOD,
                roll_pid,
                pitch_pid,
                yaw_pid
            )
            
            await drone.update(action = action,display=False)
    finally:
        simulation.close()
    

if __name__ == "__main__":
    asyncio.run(main())