import asyncio
import time
from Subsystems.Gamepad import Gamepad

from Subsystems.drone import Drone
from drone.Subsystems.data.data import Action, DataSequence
import numpy as np
from drone.Subsystems.data.data import State
from simulations.simulation import ProjectAirSimSimulation
from Subsystems.PID import PID
from drone.Controllers import Controllers, PositionController, VelocityController, AttitudeController, RateController, PIDConfig


from projectairsim import drone, types
from projectairsim.types import Pose, Vector3, Quaternion

from scipy.spatial.transform import Rotation

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
    bool_render = input("Do you want to render the scene (it runs faster without the visuals)? (y/n)").lower() == "y"
    frequency = 0
    spawn_height = input("Enter the spawn height (negative value, default is -650): ")
    spawn_height = float(spawn_height)
    log_data = input("Do you want to log data? (y/n): ").lower() == "y"
    name = "rnd_waypoint"
    if (not bool_rnd_configs):
        name = "waypoint_sky"
    rate_mode = RateController.prompt_mode()
    rate_controller = RateController(rate_mode)
    attitude_controller = AttitudeController()
    velocity_controller = VelocityController()
    position_controller = PositionController()
    gamepad = Gamepad()
    try:
        while iteration < iterations:
            iteration += 1
            simulation = ProjectAirSimSimulation()
            simulation.DATA_PATH = "data_sequences/rnd_waypoint_training/"
            simulation.configs.render = bool_render
            simulation.log_data = log_data
            simulation.local_data = False
            simulation.data_name = str(simulation.DATA_PATH + (name + str(iteration)))
            simulation.split_logs = True
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
                start_pose,start_orientation = simulation.set_pose_to_random_in_sky(height = spawn_height,rnd_pitch_and_roll=False)
                # Environment uses MSL altitude = scene origin altitude - NED Z.
                home_altitude = float(simulation.world.home_geo_point["altitude"])
                rate_controller.create(
                    simulation.configs,
                    altitude_m=home_altitude - simulation.state[State.PZ],
                )
                attitude_controller.reset()
                velocity_controller.reset()
                position_controller.reset()

                start_time = time.perf_counter()

                drone.data_sequence = None

                #Summon the "target" to fly to
                drone.state, drone.time_stamp = drone.flight_controller.state,drone.flight_controller.time
                target_reached = True
                target_yaw = None
                score = 0

                while (drone.time_stamp < max_time):

                    if target_reached or gamepad.get_button(Gamepad.Inputs.B):
                        target_position = simulation.generate_target(start_pose)
                        target_reached = False

                    target_yaw = np.arctan2(
                        target_position[State.PY] - drone.state[State.PY],
                        target_position[State.PX] - drone.state[State.PX],
                    )

                    position = State.get_position(drone.state)
                    if (simulation.target_reached(target_position,position)):
                         target_reached = True
                         score += 1
                 
                    rate_controller.update_altitude(home_altitude - drone.state[State.PZ])
                    action = Controllers.calculate_everything(
                        position_controller=position_controller,
                        velocity_controller=velocity_controller,
                        attitude_controller=attitude_controller,
                        rate_controller=rate_controller,
                        target_position=target_position,
                        target_yaw=target_yaw,
                        current_state=drone.state,
                        dt=0.01,
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
