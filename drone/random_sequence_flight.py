from enum import Enum
import time
import numpy as np
import asyncio
from Subsystems.drone import Drone

from simulations.simulation import ProjectAirSimSimulation
from Subsystems.action import ActionSequence


RANDOM_ACTION_FREQUENCY = 20
RANDOM_ACTION_LENGTH = 3  # seconds
async def main() -> None:
    simulation = ProjectAirSimSimulation()
    simulation.log_prompt()
    drone = Drone(flight_controller=simulation)
    
    try:
        simulation.start()
        takeoff = ActionSequence.generate_from_csv("takeoff.csv")
        drone.action_sequence = takeoff
        while (await drone.update()):
            pass

        random_sequence = ActionSequence.generate_random_sequence(
            start_time = drone.time_stamp,
            time = RANDOM_ACTION_LENGTH, 
            frequency = RANDOM_ACTION_FREQUENCY
        )

        drone.action_sequence = random_sequence
        while (await drone.update()):
            pass
            
    finally:
        simulation.close()



if __name__ == "__main__":
    asyncio.run(main())  # Runner for main function

