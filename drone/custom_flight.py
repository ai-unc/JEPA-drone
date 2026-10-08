from enum import Enum
import time
import numpy as np
from Subsystems.drone import Drone
import asyncio

from simulations.simulation import ProjectAirSimSimulation
from Subsystems.action import ActionSequence, Action, State

#Plays back csv flights from action_sequences
#this can later be used for training
async def main() -> None:
    simulation = ProjectAirSimSimulation()
    drone = Drone(flight_controller=simulation)
    drone.prompt_update_frequency()
    action_sequence,initial_state = ActionSequence.generate_from_csv()

    drone.action_sequence = action_sequence
    simulation.log_prompt()
    try:
        print("Original VZ: ", initial_state[State.VZ])
        simulation.start(initial_state=initial_state)
        print("Setted VZ: ", simulation.drone.get_ground_truth_kinematics()["twist"]["linear"]["z"])
        while (await drone.update()):
            pass
    finally:
        simulation.close()

if __name__ == "__main__":
    asyncio.run(main())  # Runner for main function

