import asyncio
import time

from Subsystems.drone import Drone
from drone.Subsystems.data.data import Action
import numpy as np
from drone.Subsystems.data.data import State
from simulations.simulation import ProjectAirSimSimulation
from simulations.obstacle_course import ObstacleCourse
from drone.Controllers import Controllers, VelocityController, AttitudeController, RateController


"""
Use APF algorthim to fly to the goal while avoiding obstacles
Log Vision data in corresponding CSV outputs
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
    spawn_height = None
    log_data = input("Do you want to log data? (y/n): ").lower() == "y"
    display_tensor = input("Display Vision (depth + class)? (y/n): ").lower() == "y"
    bool_vision = True
    name = "rnd_vision_obstacle"
    if (not bool_rnd_configs):
        name = "vision_obstacle"
    rate_mode = RateController.prompt_mode()
    rate_controller = RateController(rate_mode)
    attitude_controller = AttitudeController()
    velocity_controller = VelocityController()
    control_mode = Controllers.VELOCITY
    try:
        while iteration < iterations:
            iteration += 1
            simulation = ProjectAirSimSimulation()
            simulation.vision_enabled = bool_vision
            simulation.camera.display_raw = False
            simulation.camera.display_depth = False
            simulation.camera.display_tensorized = display_tensor
            simulation.configs.vision_modes = {0: False,
                                               1:  display_tensor,
                                               3: display_tensor}
            simulation.configs.render = True
            obstacle_course = ObstacleCourse()
            obstacle_course.randomize()
            simulation.obstacle_course = obstacle_course
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

                rate_controller.print_configs()

                drone.data_sequence = None

                #Summon the "target" to fly to
                drone.state, drone.time_stamp = drone.flight_controller.state,drone.flight_controller.time
                target_reached = False
                score = 0

                target_position = simulation.obstacle_course.goal #TODO VERIFY

                while (drone.time_stamp < max_time and not target_reached):

                    if simulation.target_reached(target_position, State.get_position(drone.state)):
                        target_reached = True
                        score += 1

                    current_rotation = State.get_rotation(drone.state)
                    current_yaw = np.arctan2(current_rotation[1, 0], current_rotation[0, 0])
                    
                    rate_controller.update_altitude(home_altitude - drone.state[State.PZ])
                    roll_rate, pitch_rate, yaw_rate = 0, 0, 0

                    target_velocity, travel_yaw = simulation.obstacle_avoidance_velocity()
                    rotation_matrix, collective_thrust = velocity_controller.calculate_attitudes(
                        target_velocity, State.get_velocity(drone.state), dt=0.01,
                        yaw_angle=current_yaw,
                        hover_collective=rate_controller.control_config.hover_collective,
                        gravity_magnitude=rate_controller.control_config.gravity_magnitude,
                    )
                    roll_rate, pitch_rate, _ = attitude_controller.calculate_rates(
                        rotation_matrix, current_rotation, dt=0.01,
                    )
                    yaw_error = (travel_yaw - current_yaw + np.pi) % (2*np.pi) - np.pi
                    yaw_rate = np.clip(yaw_error, -RateController.MAX_YAW_RATE,
                                       RateController.MAX_YAW_RATE)

            
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
                    if bool_vision:
                        simulation.camera.update(simulation)
            finally:
                simulation.close()
                print("SCORE: ", score)
                print("REAL TIME PASSED: ", time.perf_counter() - start_time)
    finally:
        pass

if __name__ == "__main__":
    asyncio.run(main())
