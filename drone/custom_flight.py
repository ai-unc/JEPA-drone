from enum import Enum
import time
import numpy as np
from Subsystems.drone import Drone
import asyncio

from simulations.simulation import ProjectAirSimSimulation
from simulations.obstacle_course import ObstacleCourse
from drone.Subsystems.data.data import DataSequence, Action, State

#Plays back csv flights from action_sequences
#this can later be used for training
async def main() -> None:
    simulation = ProjectAirSimSimulation()
    drone = Drone(flight_controller=simulation)
    drone.prompt_update_frequency()
    data_sequence, sim_configs = DataSequence.generate_from_csv()
    simulation.set_configs(sim_configs)
    course = ObstacleCourse()
    if course.read_csv(data_sequence.source_path):
        simulation.obstacle_course = course

    drone.data_sequence = data_sequence
    simulation.log_prompt()
    try:
        print("Original VZ: ", data_sequence.states[0][State.VZ])
        simulation.start(initial_state=data_sequence.states[0])
        print("Setted VZ: ", simulation.drone.get_ground_truth_kinematics()["twist"]["linear"]["z"])
        while (await drone.update()):
            pass
    finally:
        simulation.close()

if __name__ == "__main__":
    asyncio.run(main())  # Runner for main function

