import asyncio
import math
import numpy as np
from pathlib import Path
import json
import json5


class SimulationConfigs():
    DEFAULT_MOTOR_MASS = 0.055
    DEFAULT_FRAME_MASS = 1
    DEFAULT_TOTAL_MASS = DEFAULT_FRAME_MASS + 4 * DEFAULT_MOTOR_MASS

    DEFAULT_DRAG_COEF = 0.325
    DEFAULT_ARM_LENGTH = 0.3578
    DEFAULT_THRUST_COEF = 0.109919
    DEFAULT_TORQUE_COEF = 0.040164
    DEFAULT_MOTOR_TIME_CONSTANT = 0.005

    MIN_MASS = DEFAULT_TOTAL_MASS * 0.5
    MAX_MASS = DEFAULT_TOTAL_MASS * 1.5
    MIN_THRUST_COEF = 0.1
    MAX_THRUST_COEF = 0.28
    MIN_TORQUE_COEF = DEFAULT_TORQUE_COEF * 0.5
    MAX_TORQUE_COEF = DEFAULT_TORQUE_COEF * 1.5
    MIN_ARM_LENGTH = DEFAULT_ARM_LENGTH * 0.5
    MAX_ARM_LENGTH = DEFAULT_ARM_LENGTH * 1.5
    MIN_MOTOR_TIME_CONSTANT = DEFAULT_MOTOR_TIME_CONSTANT * 0.5
    MAX_MOTOR_TIME_CONSTANT = DEFAULT_MOTOR_TIME_CONSTANT * 1.5

    DEFAULT_MOTOR_Z = -0.01

    MAX_RPM : np.double = 6396.667

    CSV_PREFIX : str = "# CONFIG: "
    SIMULATION_FOLDER = Path(__file__).resolve().parent
    CONFIG_PATH  = SIMULATION_FOLDER / "custom_sim_config"
    CONFIG_FILE = CONFIG_PATH / "custom_config.jsonc"

    @staticmethod
    def calculate_motor_positions(arm_length: float) -> list[np.ndarray]:
        offset = arm_length / math.sqrt(2)
        return [
            np.round(np.array([ offset,  -offset, SimulationConfigs.DEFAULT_MOTOR_Z]), 7), #FL
            np.round(np.array([ offset,  offset, SimulationConfigs.DEFAULT_MOTOR_Z]), 7), #FR
            np.round(np.array([-offset,  -offset, SimulationConfigs.DEFAULT_MOTOR_Z]), 7), #BL
            np.round(np.array([-offset, offset, SimulationConfigs.DEFAULT_MOTOR_Z]), 7) #BR
        ]

    def __init__(self):
        self.mass = self.DEFAULT_TOTAL_MASS
        self.thrust_coef = self.DEFAULT_THRUST_COEF
        self.torque_coef = self.DEFAULT_TORQUE_COEF
        self.arm_length = self.DEFAULT_ARM_LENGTH
        self.motor_time_constant = self.DEFAULT_MOTOR_TIME_CONSTANT
        self.drag_coef = self.DEFAULT_DRAG_COEF

        self.motor_positions = self.calculate_motor_positions(self.arm_length)
        self.path = self.CONFIG_PATH
        self.file = self.CONFIG_FILE

    def randomize(self) -> None:
        print("Randomizing configs")
        #Round each to 7 digits
        self.mass = round(np.random.uniform(self.MIN_MASS, self.MAX_MASS), 7)
        self.thrust_coef = round(np.random.uniform(self.MIN_THRUST_COEF, self.MAX_THRUST_COEF), 7)
        self.torque_coef = round(np.random.uniform(self.MIN_TORQUE_COEF, self.MAX_TORQUE_COEF), 7)
        self.arm_length = round(np.random.uniform(self.MIN_ARM_LENGTH, self.MAX_ARM_LENGTH), 7)
        self.motor_time_constant = round(np.random.uniform(self.MIN_MOTOR_TIME_CONSTANT, self.MAX_MOTOR_TIME_CONSTANT), 7)
        self.drag_coef = round(np.random.uniform(0, self.DEFAULT_DRAG_COEF), 7)

        self.motor_positions = self.calculate_motor_positions(self.arm_length)

    def to_string(self) -> str:
        return (
            f"SimulationConfigs: "
            f"config path={self.path}, "
            f"mass={self.mass}, "
            f"thrust_coef={self.thrust_coef}, "
            f"torque_coef={self.torque_coef}, "
            f"arm_length={self.arm_length}, "
            f"motor_time_constant={self.motor_time_constant}, "
            f"drag_coef={self.drag_coef}"
        )

    def config_to_csv(self) -> str:
        config = {
            "mass": self.mass,
            "thrust_coef": self.thrust_coef,
            "torque_coef": self.torque_coef,
            "arm_length": self.arm_length,
            "motor_time_constant": self.motor_time_constant,
            "drag_coef": self.drag_coef
        }

        return self.CSV_PREFIX + json.dumps(config)

    @staticmethod
    def csv_to_config(config_values: str) -> "SimulationConfigs":
        # Remove optional CSV comment prefix
        config_values = config_values.strip()

        if config_values.startswith(SimulationConfigs.CSV_PREFIX):
            config_values = config_values[len(SimulationConfigs.CSV_PREFIX):].strip()

        values = json.loads(config_values)

        config = SimulationConfigs()
        config.mass = float(values["mass"])
        config.thrust_coef = float(values["thrust_coef"])
        config.torque_coef = float(values["torque_coef"])
        config.arm_length = float(values["arm_length"])
        config.motor_time_constant = float(values["motor_time_constant"])
        config.drag_coef = float(values["drag_coef"])

        config.motor_positions = config.calculate_motor_positions(config.arm_length)

        return config

    def write(self) -> None:
        """
        Apply physics parameters to a Project AirSim robot JSONC file.

        Must be called before the robot is loaded into the simulation.
        """
        print("Writing Configs: ", self.to_string())

        config_file = Path(self.file)

        # Read the original Project AirSim configuration
        with config_file.open("r", encoding="utf-8") as f:
            config = json5.load(f)

        # Recalculate positions in case arm_length changed
        self.motor_positions = self.calculate_motor_positions(
            self.arm_length
        )

        # Scale masses to match desired total mass
        mass_multiplier = self.mass / self.DEFAULT_TOTAL_MASS

        motor_names = ["Prop_FL", "Prop_FR", "Prop_RL", "Prop_RR"]

        motor_positions = {
            name: pos
            for name, pos in zip(motor_names, self.motor_positions)
        }

        def position_string(position):
            return " ".join(str(float(value)) for value in position)

        # Update frame and motor links
        for link in config["links"]:
            name = link["name"]
            inertial = link["inertial"]

            # Mass
            if name == "Frame":
                inertial["mass"] = (
                    self.DEFAULT_FRAME_MASS * mass_multiplier
                )
            elif name in motor_positions:
                inertial["mass"] = (
                    self.DEFAULT_MOTOR_MASS * mass_multiplier
                )

            # Aerodynamic drag
            inertial["aerodynamics"]["drag-coefficient"] = (
                float(self.drag_coef)
            )

            # Motor physical positions
            if name in motor_positions:
                xyz = position_string(motor_positions[name])

                inertial["origin"]["xyz"] = xyz
                link["visual"]["origin"]["xyz"] = xyz

        # Update rotor actuators
        for actuator in config["actuators"]:
            if actuator["type"] != "rotor":
                continue

            name = actuator["child-link"]

            # Motor force application position
            if name in motor_positions:
                actuator["origin"]["xyz"] = position_string(
                    motor_positions[name]
                )

            settings = actuator["rotor-settings"]

            settings["coeff-of-thrust"] = float(self.thrust_coef)
            settings["coeff-of-torque"] = float(self.torque_coef)
            settings["smoothing-tc"] = float(
                self.motor_time_constant
            )
            settings["max-rpm"] = float(self.MAX_RPM)
        print("SUCCESSFULLY WROTE CONFIG TO", config_file)

        # Save as JSON (also valid JSONC)
        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        