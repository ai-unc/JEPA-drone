# SUMMARY (11/6/26)
This read me contains info on how to setup the simulation, how the simulation works, the format data is logged in, and how data is acquired.

# FYI (10/7/26)
The contents of the drone folder may be moved up outside of the drone folder and then the drone folder deleted if that makes training JEPA easier, I have all my (Ethan) simulation and control stuff in this "drone" folder because I did not want to be intrusive on the project.

# SIMULATION SETUP (10/7/26)
To run the simulator you need to run the server and the client.
The server is not on this file, you have to download it (see step 1).
Once you have the server running, you need to run a client. There are some sample python clients in this folder:

custom_flights.py lets you chose a csv file in data_sequences to run. This is good for replaying flights
gamepad_flights.py lets you manually use a controller to fly a drone.

waypoint_flight.py spawns a drone in and has the drone autonomously move to various points. Good for autonomous data acquisition.

All flights can be logged, and if chosen to, will show up in flight_logs. If you want to replay them, copy or move it to data_sequences, and then run them through custom_flights.py.


# Step 1. Setup server: Download an Environment
-> I am going to use blocks:
https://github.com/iamaisim/ProjectAirSim/releases

-> On version 1.01 there is Blocks-Windows-1.0.1.zip 
-> Download that
-> Extract it somewhere
-> Open it and you should see an Unreal Engine app called Blocks
-> Open it. If windows pops up, clicck more info then run anyway
-> Setup Unreal Engine, and then keep it open.

IMPORTANT: If the simulator ever fails to run, try using task manager and completely killing it.

# Step 2. Setup client: Done through uv
This is done through uv, so if you followed the uv instructions you are good to go.
Here are the dependencies the simulator and client require:
```
uv add projectairsim==1.0.2
uv add pandas
uv add sciPy
uv add json5
```
# Step 3. Run the client
-> Go to the drone
```
cd drone
uv run python custom_flight.py
```
TODO: I might make this be able to be ran outside of the drone folder

# Other info that is useful:
For simulation I have time be steppable, not real time, this way lag does not effect anything
I do this by having drone/simulations/sim_config/scene_basic_drone.jsonc have this:
```
"clock": {
    "type": "steppable",
    "step-ns": 1000000,
    "real-time-update-rate": 3000000,
    "pause-on-start": true
  },
```
step-ns is typically 3000000 but I have it as 1000000 because that divides nicely with 10 miliseconds (100Hz) compared to 3 miliseconds -> floating point error sucks man

Anyways because its steppable I run this in the simulation:
```
self.world.continue_for_n_steps(
    FlightController.CONTROL_STEPS_PER_MILLISECOND,
    wait_until_complete=True,
)
```

where CONTROL_STEPS_PER_MILLISECOND is 10 because 1 step is 1 milliseconds and 100 Hz is 10 milliseconds

Also I have the simulator act as a flight controller because they are essentially doing the same thing relative to the drone:
1. Send measured state to the drone.
2. Recieve actions and do said actions which causes a corresponding change of measured state.
3. Allows for easy logging of both flight controller data and simulator data.

# Drone/Simulation Configs
Currently the following variables are optionally randomized:

```
mass
thrust_coef
torque_coef
arm_length
motor_time_constant
drag_coef
```
See simulation_config.py for the variation and default values

# Data logs
The csvs have info as follows:
- Header: Drone/Simulation Configs 
- Rows: time,FL,FR,BL,BR,x,y,z,vx,vy,vz,r1x,r1y,r1z,r2x,r2y,r2z,r3x,r3y,r3z,wx,wy,wz

*Example header: # CONFIG: {"mass": 1.22, "thrust_coef": 0.109919, "torque_coef": 0.040164, "arm_length": 0.3578, "motor_time_constant": 0.005, "drag_coef": 0.325}*

*time is seconds passed since the start of simulation.* 

*FL,FR,BL,BR are motor actions (0 - 4.18 where 4.18 is assumed max thrust).*

*x,y,z are position, vx,vy,vz are velocity, r1x,r1y,r1z...r3z is the rotation matrix. wx,wy,wz are the angular velocities.*


# Flight Modes (11/6/26)

The drone has the following flight modes:
1. Motor Thrusts (fl,fr,bl,br)
2. Rate (Angular Velocities) + Collective thrust. 
- Inputs: Target pitch/roll/yaw rates, current pitch/roll/yaw rates, collective thrust
- Outputs: Target Motor thrusts
3. Attitude (Angles) + Collective Thrust. 
- Inputs: Target Rotation matrix, Current Rotation matrix, Collective thrust
- Outputs: Target Rates
4. Velocity 
- Inputs: Target Velocity, Current Velocity
- Outputs: Target Attitude + Collective Thrust
5. Position Autonomous only
- Inputs: Target Position, Current Position
- Outputs: Velocity



Everything is in ascending order of abstraction where:
Position control needs velocity control which needs attitude control which needs rate control which needs motor control

All the controllers utilize a series of PID controllers + some physics to generate output

# Autotuner (11/6/26)
Codex generated a physics model that automatically creates the gains (P in PID) for the rate controller. The layers above the rate controller do not need to change assuming the rate controller works.

See a video on it here:
https://www.youtube.com/watch?v=E7-F3rcVzWI

# Autonomous Flight & Data Collection (11/6/26)
Since the drone can be running random configurations and, using the autotuner, can fly to a target point, this is useful for data collection.

On my Computer (Ethan) collecting data with the simulation rendering is about 2x real life time. So for 20k 10 second flights, that would take approximately 55 hours. To make it faster I am making some sacrifices:

1. Since reloading the simulation takes about 5 seconds, I am making the flights 100 seconds long and doing 2k flights, that way it has to reload less. This then gets split up into 10 second flights.
2. I am disabling rendering for increased simulation speed.

