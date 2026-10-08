from enum import Enum
import math
import time
import numpy as np
import asyncio
from Subsystems.drone import Drone

from simulations.simulation import ProjectAirSimSimulation
from Subsystems.action import Action, ActionSequence


RANDOM_ACTION_LENGTH = 10  # seconds
async def main() -> None:
    iteration = int(input("Enter the starting iteration: "))
    iterations = input("How many times do you want to run this sequence? ")
    iterations = int(iterations) + iteration
    name = "rnd_sky"

    while iteration < iterations:
        iteration += 1
        simulation = ProjectAirSimSimulation()
        simulation.log_data = True
        simulation.data_name = str(simulation.DATA_PATH + (name + str(iteration)))
        drone = Drone(flight_controller=simulation)
        #Max speed
        drone.update_frequency = 0
        RANDOM_ACTION_FREQUENCY = np.random.uniform(1,100)
        thrust_range1 = np.random.uniform(0, Action.ASSUMED_MAX_THRUST)
        thrust_range2 = np.random.uniform(0, Action.ASSUMED_MAX_THRUST)
        max_thrust = max(thrust_range1, thrust_range2)
        min_thrust = min(thrust_range1, thrust_range2)

        print("ITERATION " + str(iteration))
        print("RANDOM ACTION FREQUENCY: " + str(RANDOM_ACTION_FREQUENCY))
        print("THRUST RANGE: " + str(min_thrust) + " - " + str(max_thrust))

        try:
            simulation.start()
            simulation.set_pose_to_random_in_sky(height = -350)
            random_sequence = ActionSequence.generate_random_sequence(
                start_time = drone.time_stamp,
                time = RANDOM_ACTION_LENGTH, 
                frequency = RANDOM_ACTION_FREQUENCY,
                thrust_variance = (min_thrust, max_thrust)
            )

            drone.action_sequence = random_sequence
            while (await drone.update()):
                pass
                
        finally:
            simulation.close()



if __name__ == "__main__":
    asyncio.run(main())  # Runner for main function

