# FYI (10/7/26)
The contents of the drone folder may be moved up outside of the drone folder and then the drone folder deleted if that makes training JEPA easier, I have all my (Ethan) simulation and control stuff in this "drone" folder because I did not want to be intrusive on the project.

# SUMMARY (10/7/26)
To run the simulator you need to run the server and the client.
The server is not on this file, you have to download it (see step 1).
Once you have the server running, you need to run a client. There are some sample python clients in this folder:

custom_flights.py lets you chose a csv file in action_sequences to run. This is good for replaying flights
gamepad_flights.py lets you manually use a controller to fly a drone.
random_sequence_flight.py flies up and then does a random sequence of motor outputs.

All flights can be logged, and if chosen to, will show up in flight_logs. If you want to replay them, copy or move it to action_sequences, and then run them through custom_flights.py.

In the future there will definetly be a JEPA related script.

TODO -> Make randomization of variables for the simulator so that JEPA can be trained on non-deterministic actions.

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

For logging data I have it log both action and state which may or may not be helpful for training.
