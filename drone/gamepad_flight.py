import asyncio
import time
from Subsystems.Gamepad import Gamepad

from Subsystems.drone import Drone
from drone.Subsystems.data.data import Action, DataSequence
import numpy as np
from drone.Subsystems.data.data import State
from simulations.simulation import ProjectAirSimSimulation
from Subsystems.PID import PID
from drone.Controllers import Controllers, AttitudeController, RateController, PIDConfig


from projectairsim import drone, types
from projectairsim.types import Pose, Vector3, Quaternion

from scipy.spatial.transform import Rotation

# Separate gamepad entry point using pid_autotuner.py.
#There is a bunch of AI Generated code here. This was used to fly around and collect data.
#To be flyable by a human there needs to be several PID controllers to control rotational velocity
#Which is why this file is so massive


"""
The idea: run the simulator at the maximum speed the computer can handle.
Have the flight last 100 in simulator seconds, which is split into 10, 10 second flight logs.
Assuming simulation configs are randomized, have an auto tuner for PID controls based on sim configs.
Since flight is sped up, acro control would be to hard, so rotation and velocity control is ideal.
So I dont get bored flying, a "target" will spawn within a certain radius of my drone.
Since I am lazy and there is no way to disable collisions in this python version,
I will make the target teleport somewhere else whenever you get too close.
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
                simulation.set_pose_to_random_in_sky(height = spawn_height)
                rate_controller.create(simulation.configs)
                attitude_controller.reset()
                start_time = time.perf_counter()
                for asset in simulation.world.list_assets(
                                ".*[Ss]phere.*|.*[Cc]ube.*|.*[Cc]one.*|.*[Cc]ylinder.*"
                            ):
                                print(asset)



                rate_controller.print_configs()

                drone.data_sequence = None

                #Summon the "target" to fly to
                drone.state, drone.time_stamp = drone.flight_controller.state,drone.flight_controller.time
                rng = np.random.default_rng()
                target_reached = True
                target_name = None
                target_pose = None
                score = 0

                while (drone.time_stamp < max_time):

                    if target_reached or gamepad.get_button(Gamepad.Inputs.B):
                        target_reached = False
                        angle = rng.uniform(0, 2 * np.pi)
                        distance = rng.uniform(50, 200)

                        # World position in metres. Negative Z is above ground.
                        target_pose = types.Transform({
                            "translation": {
                                "x": float(
                                    drone.state[State.PX] + distance * np.cos(angle)
                                ),
                                "y": float(
                                    drone.state[State.PY] + distance * np.sin(angle)
                                ),
                                "z": float(min(
                                    -5.0,
                                    drone.state[State.PZ] + rng.uniform(-10, 10),
                                )),
                            },
                            # Identity quaternion: no rotation.
                            "rotation": {
                                "w": 1.0,
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                            },
                        })

                        if target_name is None:
                            target_name = simulation.world.spawn_object(
                                object_name="FlightTarget",
                                asset_path="Sphere",
                                object_pose=target_pose,
                                object_scale=[8.0, 8.0, 8.0],
                                enable_physics=False,
                            )
                            if not target_name:
                                raise RuntimeError("Failed to spawn FlightTarget.")
                        else:
                            simulation.world.set_object_pose(
                                object_name=target_name,
                                object_pose=target_pose,
                                teleport=True,
                            )

                    target_position = np.array([
                        target_pose["translation"]["x"],
                        target_pose["translation"]["y"],
                        target_pose["translation"]["z"],
                    ], dtype=float)

                    x,y,z = State.get_position(drone.state)
                    tx,ty,tz = target_position[0],target_position[1],target_position[2]
                    distance_from_target = State.get_distance(x,tx,y,ty,z,tz)
                    if (distance_from_target < 20):
                         target_reached = True
                         score += 1

                    throttle,roll_rate,pitch_rate,yaw_rate = 0,0,0,0

                    if control_mode == Controllers.ATTITUDE:
                        throttle,roll,pitch,yaw_rate = gamepad.attitude_control_inputs()
                        rotation_matrix = attitude_controller.gamepad_to_attitude(roll,pitch,State.get_rotation(drone.state))
                        roll_rate,pitch_rate,_ = attitude_controller.calculate_rates(rotation_matrix,State.get_rotation(drone.state),dt = 0.01)
                    elif control_mode == Controllers.RATE:
                        throttle,roll_rate,pitch_rate,yaw_rate = gamepad.rate_control_inputs()

                    if control_mode == Controllers.ATTITUDE:
                        # Attitude outputs are already rates in rad/s; only yaw is a stick input.
                        collective_thrust = throttle * Action.ASSUMED_MAX_THRUST
                        yaw_rate *= RateController.MAX_YAW_RATE
                    else:
                        throttle *= rate_controller.control_config.throttle_sensitivity
                        roll_rate *= rate_controller.control_config.roll_rate_sensitivity
                        pitch_rate *= rate_controller.control_config.pitch_rate_sensitivity
                        yaw_rate *= rate_controller.control_config.yaw_rate_sensitivity
                        collective_thrust,roll_rate,pitch_rate,yaw_rate = rate_controller.gamepad_to_target_rates(
                            throttle,roll_rate,pitch_rate,yaw_rate
                        )


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
