import asyncio
import time
from Subsystems.Gamepad import Gamepad

from Subsystems.drone import Drone
from drone.Subsystems.data.data import Action
import numpy as np
from drone.Subsystems.data.data import State
from simulations.simulation import ProjectAirSimSimulation
from drone.Controllers import Controllers, VelocityController, AttitudeController, RateController



# Separate gamepad entry point using pid_autotuner.py.
#There is a bunch of AI Generated code here. This was used to fly around and collect data.
#To be flyable by a human there needs to be several PID controllers to control rotational velocity
#Which is why this file is so massive


"""
The idea:
1. Achieve rate control via an autotuner
2. Achieve altitude control that uses the rate controller and doesn't need to be autotuned
3. Achieve velocity control that uses attitude control and doesn't need to be autotuned
Once those 3 steps are achieved, autonomous position control is next.
With autonomous position control, massive data collection is possible.
--Ethan
"""
async def main():
    max_time = float(input("Enter the maximum simulation time: "))
    iteration = int(input("Enter the starting iteration: "))
    iterations = input("How many times do you want to run this sequence? ")
    iterations = int(iterations) + iteration
    bool_rnd_configs = input("Do you want to use random configurations? (y/n): ").lower() == "y"
    frequency = input("Enter the update frequency in seconds (real time is 100, as fast as possible is 0, 50 percent is 50): ")
    frequency = float(frequency)
    spawn_height = input("Enter the spawn height (negative value, default is -650): ")
    spawn_height = float(spawn_height)
    log_data = input("Do you want to log data? (y/n): ").lower() == "y"
    name = "rnd_gamepad"
    if (not bool_rnd_configs):
        name = "gamepad_sky"
    rate_mode = RateController.prompt_mode()
    rate_controller = RateController(rate_mode)
    attitude_controller = AttitudeController()
    velocity_controller = VelocityController()
    control_mode = Controllers.prompt_mode()
    gamepad = Gamepad()
    try:
        while iteration < iterations:
            iteration += 1
            simulation = ProjectAirSimSimulation()
            simulation.log_data = log_data
            simulation.local_data = False
            simulation.data_name = str(simulation.DATA_PATH + (name + str(iteration)))
            if (bool_rnd_configs):
                simulation.configs.randomize()
            simulation.configs.write()
            drone = Drone(flight_controller=simulation)
            drone.update_frequency = frequency

            action = None

            score = 0
            start_time = time.perf_counter()
            try:
                simulation.start()
                start_position, _ = simulation.set_pose_to_random_in_sky(
                    height=spawn_height, rnd_pitch_and_roll=False,
                )
                # Environment uses MSL altitude = scene origin altitude - NED Z.
                home_altitude = float(simulation.world.home_geo_point["altitude"])
                rate_controller.create(
                    simulation.configs,
                    altitude_m=home_altitude - simulation.state[State.PZ],
                )
                attitude_controller.reset()
                velocity_controller.reset()

                start_time = time.perf_counter()
                for asset in simulation.world.list_assets(
                                ".*[Ss]phere.*|.*[Cc]ube.*|.*[Cc]one.*|.*[Cc]ylinder.*"
                            ):
                                print(asset)



                rate_controller.print_configs()

                drone.data_sequence = None

                #Summon the "target" to fly to
                drone.state, drone.time_stamp = drone.flight_controller.state,drone.flight_controller.time
                target_reached = True
                score = 0

                while (drone.time_stamp < max_time):

                    if target_reached or gamepad.get_button(Gamepad.Inputs.B):
                        target_position = simulation.generate_target(start_position)
                        target_reached = False

                    if simulation.target_reached(target_position, State.get_position(drone.state)):
                        target_reached = True
                        score += 1

                    current_rotation = State.get_rotation(drone.state)
                    current_yaw = np.arctan2(current_rotation[1, 0], current_rotation[0, 0])
                    
                    rate_controller.update_altitude(home_altitude - drone.state[State.PZ])
                    throttle,roll_rate,pitch_rate,yaw_rate = 0,0,0,0
                    if control_mode == Controllers.VELOCITY:
                        vertical_input, horizontal_input, forward_input, yaw_rate = gamepad.velocity_control_inputs()
                        # Scale heading-relative inputs before rotating into world axes.
                        forward_velocity = forward_input * VelocityController.MAX_X_VELOCITY
                        horizontal_velocity = horizontal_input * VelocityController.MAX_Y_VELOCITY
                        vertical_velocity = vertical_input * VelocityController.MAX_Z_VELOCITY
                        world_x, world_y = VelocityController.relative_to_world(
                            forward=forward_velocity,
                            horizontal=horizontal_velocity,
                            yaw=current_yaw,
                        )
                        target_velocity = np.array([world_x, world_y, vertical_velocity], dtype=float)
                        # Yaw is commanded separately as a body rate. Preserve
                        # current heading in the velocity target so the attitude
                        # error does not couple an uncommanded yaw turn into tilt.
                        rotation_matrix,collective_thrust = velocity_controller.calculate_attitudes(
                            target_velocity, State.get_velocity(drone.state), dt=0.01,
                            yaw_angle=current_yaw,
                            hover_collective=rate_controller.control_config.hover_collective,
                            gravity_magnitude=rate_controller.control_config.gravity_magnitude,
                        )
                        roll_rate,pitch_rate,_ = attitude_controller.calculate_rates(
                            rotation_matrix, current_rotation, dt=0.01,
                        )
                    elif control_mode == Controllers.ATTITUDE:
                        throttle,roll,pitch,yaw_rate = gamepad.attitude_control_inputs()
                        collective_thrust = throttle * Action.ASSUMED_MAX_THRUST * rate_controller.control_config.throttle_sensitivity
                        rotation_matrix = attitude_controller.gamepad_to_attitude(roll,pitch,State.get_rotation(drone.state))
                        roll_rate,pitch_rate,_ = attitude_controller.calculate_rates(rotation_matrix,State.get_rotation(drone.state),dt = 0.01)
                    elif control_mode == Controllers.RATE:
                        throttle,roll_rate,pitch_rate,yaw_rate = gamepad.rate_control_inputs()
                        collective_thrust = throttle * Action.ASSUMED_MAX_THRUST * rate_controller.control_config.throttle_sensitivity


                    yaw_rate *= RateController.MAX_YAW_RATE * rate_controller.control_config.yaw_rate_sensitivity

                    #Only apply pitch and roll rate sensitivity when those are directly controlled
                    if control_mode == Controllers.RATE:
                        roll_rate = roll_rate * RateController.MAX_ROLL_RATE * rate_controller.control_config.roll_rate_sensitivity
                        pitch_rate = pitch_rate * RateController.MAX_PITCH_RATE * rate_controller.control_config.pitch_rate_sensitivity
                    elif control_mode == Controllers.VELOCITY:
                         yaw_rate *= 0.5 #Control preference since yaw rate seems a lot more powerful than the velocity control

                    angular_velocity = State.get_angular_velocity(drone.state)
                    action = rate_controller.calculate_action(
                        angular_velocity=angular_velocity,
                        collective_thrust=collective_thrust,
                        target_roll_rate=roll_rate,
                        target_pitch_rate=pitch_rate,
                        target_yaw_rate=yaw_rate,
                        dt = 0.01
                    )

                    await drone.update(action = action,display=False)
            finally:
                simulation.close()
                print("SCORE: ", score)
                print("REAL TIME PASSED: ", time.perf_counter() - start_time)
    finally:
        gamepad.stop()


if __name__ == "__main__":
    asyncio.run(main())
