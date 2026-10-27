# SkyJEPA

This README provides a comprehensive overview of the entire project and how everything wires together.
Refer here for any questions about design decisions, information flow, etc.

Below is a helpful visualization of the original SkyJEPA paper (thanks Ethan!)

### Visualization of SkyJEPA
![Ethan's visualization](./images/SkyJEPA_visualization.png)

## Breakdown of Folders

### /skyjepa (WIP)
Includes the model, the data pipelines that take CSV to input data, and the training steps.

### /drone (WIP)
Includes the ProjectAirSim client, flight control, and data logging (simulation to CSV).

### /docs
General documentation. Contains tensor_shapes.md which I'm pretty sure is out of date.

### /images
Contains image files, including visualizations of the project and model structure, as well as future output images from intermediate and final steps in the project.

## Breakdown of Files by Section

### /skyjepa

- **/data**
  Turns simulator CSVs into the four window tensors the models train on (`state_history`, `action_history`, `future_states`, `future_actions`).
  - **dataset.py**
    Slides a history window of length H and a future horizon of length T along every trajectory and stacks the windows.
  - **trajectory_loader.py**
    Reads the 100 Hz flight CSVs (default: `drone/data_sequences/rnd_sky_training`), downsamples to the model `dt`, and can hold out whole trajectories for evaluation.
- **/logging**
  Training progress logger used by the JEPA and baseline loops.
  - **logger.py**
    Prints tagged `[INFO]` / `[WARNING]` / `[ERROR]` lines and helpers for timing GPU work.
- **/models**
  The JEPA networks: two history encoders plus the latent dynamics predictor.
  - **encoders.py**
    `StateHistoryEncoder` and `ActionHistoryEncoder`: TCNs that map an H-step window to a `latent_dim` vector.
  - **predictor.py**
    `LatentPredictor`: GRUCell that maps `(s_t, z_t)` to `s_{t+1}` and rolls that out over the horizon.
- **__init__.py**
  Package entry. Re-exports `Config`.
- **baseline.py**
  Direct state-prediction baseline: one MLP maps recent state/action history to the next physical state, then rolls that out for T steps so it can be compared with the JEPA on the same windows.
- **config.py**
  Shared sizes and training defaults (`state_dim`, `action_dim`, `latent_dim`, history H, horizon T, `dt`, batch size, VICReg weights, etc.). Tensor layouts that go with these numbers are in `docs/tensor_shapes.md`.
- **losses.py**
  VICReg latent prediction loss (invariance + variance + covariance) plus collapse checks on the latents.
- **run_baseline.py**
  CLI to train the baseline on drone logs, hold out whole trajectories, and write error-vs-horizon plots/tables under `artifacts/`.
- **run_jepa.py**
  CLI to load a saved `artifacts/jepa.pt` checkpoint (encoders, predictor, normalizer) for eval.
- **train.py**
  Training entry point. Loads trajectories, builds windows, then trains either the JEPA or the baseline (`python -m skyjepa.train jepa` / `baseline`). Writes `artifacts/jepa.pt` for the JEPA.
- **windows.py**
  Cuts overlapping H-step clips out of the four window tensors for each JEPA rollout step.

### /drone

NOTE: you can check out Ethan's README inside of the drone directory, please check that out for a more detailed description on use. This part also requires a download of additional server software and packages through uv, so please refer to that walkthrough for the most up to date instructions.

OUTSTANDING QUESTIONS: What actions are acceptable in the random-thrust flights? In about 50% of them, the drone goes upside down (is the drone physically capable of flying upside down in the real world and/or sim, and should that be something that we're training on?)

CSV write spec (N, 23): (time(s), FL(N), FR(N), BL(N), BR(N), x(m), y(m), z(m), vx(m/s), vy(m/s), vz(m/s), r1x, r1y, r1z, r2x, r2y, r2z, r3x, r3y, r3z, wx(rad/s), wy(rad/s), wz(rad/s))

Where each timestamp progresses in discrete 10ms steps, the next 4 parameters are inputs to the four rotors in Newtons, and the 9 rXX correspond to individual entries in the 3x3 rotation matrix of the drone.

- **/data_sequences**
  - **/rnd_sky_training**
    Contains 200 files of random-thrust 10s flights, this is the base on which we plan to train JEPA. This uses standard physics (ie. none of the six factors in the SimulationConfigs was randomized).
  - **/rnd_configs_rnd_action_training**
    Contains 200 files of random-thrust 10s flights, using a different random physics set in each file (ie. the six factors in the SimulationConfigs were randomized in the prescribed range by the paper). NOTE: some files have blow up, so we should look into it.
  - **/rnd_gamepad_training**
    28 10s manually controlled flights, with random physics.
  - **/gamepad_training**
    2 10s manually controlled flights, with default physics.
- **/simulations**
  placeholder
- **/subsystems**
  placeholder
- **custom_flight.py**
- **gamepad_flight.py**
- **projectairsim_client.log**
- **random_sequence_flight.py**
