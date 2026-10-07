from enum import Enum
import time
import numpy as np
from Subsystems.drone import Drone
import asyncio

from simulations.simulation import ProjectAirSimSimulation
from Subsystems.action import Action, Motor, ActionSequence
from Subsystems.state import State

async def main() -> None:
    simulation = ProjectAirSimSimulation()
    drone = Drone(flight_controller=simulation)
    action_sequence = ActionSequence.generate_from_csv()

    drone.action_sequence = action_sequence
    simulation.log_prompt()
    try:
        simulation.start()
        while (await drone.update()):
            pass
    finally:
        simulation.close()

if __name__ == "__main__":
    asyncio.run(main())  # Runner for main function

