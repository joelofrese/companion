# Companion Drone

This is a living guide. Keep it short, current, and editable. Record only
decisions or evidence that future work needs; delete stale history.

## Purpose

Build a slow, natural indoor companion drone. The AI chooses behavior; lower
layers turn that behavior into safe motion. It should feel creature-like, not
like a point-to-point robot.

## Working rules

- Prefer the simplest complete design. Readability and clear control flow
  outrank feature speed.
- Remove obsolete code, tests, comments, wrappers, and compatibility paths.
  Add files, dependencies, and abstractions only for present value.
- Work autonomously. Read `STEERING.md` once at each meaningful milestone
  before selecting work; do not poll it during implementation.
- Make the smallest complete vertical slice, then simplify the final diff.
  Preserve a working baseline before substantial restructuring.
- Preserve unrelated user changes. Work on a feature branch; make verified,
  cohesive, reversible checkpoints and push stable ones when possible.
- If hardware or credentials block one task, continue with useful simulation.
  Never weaken safety to make progress.

## Safety architecture

The layers communicate only with the layer immediately below:

| Layer | Runs on | Responsibility |
|---|---|---|
| Cognitive | Mac | LLM, vision, voice, and slow state transitions |
| Reactive | Mac | Converts state and perception into bounded velocity commands |
| Stabilization | PX4 | Attitude and motor control |

The Mac sends only velocity intent. The CM5 independently rejects malformed,
stale, reordered, or out-of-bounds UDP commands and lets a fresh forward
obstacle reading override them. PX4 receives only velocity setpoints; no layer
sends motors or absolute positions.

The state model is `IDLE → FOLLOWING → AVOIDING → HOVERING → RESPONDING → IDLE`.
Cognitive intent owns persistent state. Obstacle safety can temporarily report
`AVOIDING` and command bounded backoff, then restore the saved intent. Normal
visual following closes distance or holds; only obstacle safety reverses.

## System boundaries

- Drone: DroneBlox DEXI 3 with PX4 (FMUv6X), Raspberry Pi CM5, optical flow,
  forward TOF, and Pi camera.
- Mac: all ML and high-level control. CM5: sensor/video relay and final command
  safety/forwarding. Link: local Wi-Fi.
- Follow geometry uses tracked target height for north correction and horizontal
  image error for east correction. Kalman prediction compensates for delay.
- Push-to-talk Whisper is the initial voice interface. Unknown or conflicting
  speech does nothing; explicit stop/hover is safe.
- Use DEXI-OS. Do not modify power electronics or add a tether. Hardware flight
  is brief and follows simulation evidence.

## Validation

- Tests protect distinct observable behavior, safety properties, or active
  integration contracts. Delete redundant implementation-coupled tests.
- Use bounds and tolerances for tunable flight behavior. Unit tests are fast
  evidence; PX4 SITL/Gazebo is the authority for flight behavior.
- Confirm simulator output and telemetry, not process survival. A full pass
  must show connection, readiness, arm, offboard motion, obstacle backoff,
  landing, and disarm.
- Fix failures at their source. Do not skip checks, weaken assertions, or add
  retries merely to pass.

Useful checks from `companion/`:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m unittest discover -s tests -q
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice tests
python -m sim.command_loopback
python -m sim.world
python -m sim.run_world
python -m sim.run_world --image .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg
python -m sim.offboard_full .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg
```

Start PX4 manually with `stdbuf -oL -eL make px4_sitl gz_x500` from
`~/Code/Croppie/PX4-Autopilot`, or use `python -m sim.run_world`. Wait for
`pxh>` before a verifier. If connection hangs, check UDP 14580; clean orphaned
PX4/Gazebo/MAVSDK processes before retrying.

## Hardware operation

- Mac: `python -m control.companion <cm5-ip>`. It defaults to IDLE; use
  `--state following` or one deliberate `--voice-once` utterance.
- CM5 camera: `python -m onboard.video_sender <mac-ip>`.
- CM5 ROS bridge: `python -m onboard.ros2_bridge`. It owns PX4 velocity
  setpoints and must not run beside `px4_offboard_manager`.
- Before DEXI-OS bring-up, run `python -m onboard.bringup_check`. ROS 2,
  `px4_msgs`, Micro-ROS serial, `libcamerasrc`, `/dev/ttyAMA2`, the physical
  distance topic, and Wi-Fi streaming remain hardware gates.

## Current state

- PX4 SITL/Gazebo, MAVSDK offboard velocity control, state/reactive control,
  YOLO/Kalman tracking, RTP/H.264, voice intent, UDP safety, CM5 service/ROS
  seams, and full-stack software SITL are implemented.
- The Mac path is `CompanionRuntime.tick`: vision → intent → reactive command
  → watchdog → UDP. The CM5 independently validates and forwards safe commands.
- `sim.world` is the autonomous behavioral harness. Synthetic target truth
  derives image geometry from actual PX4 NED position and drives follow, lateral
  motion, target loss, obstacle, malformed sensor, dropout, invalid command,
  recovery, zero shutdown, landing, and disarm through the real path. It is not
  a claim of camera or TOF realism.
- `sim.run_world` starts and cleans up PX4/Gazebo around the behavioral world;
  pass `--image` to run the production RTP/YOLO full-stack scenario instead.
- Hardware remains pending: CM5 Wi-Fi video, DEXI ROS serial/PX4 response, TOF
  topic/wiring, and the first flight.
- One forward TOF sensor supports stopping/backoff, not full directional
  avoidance. Do not add sensing until hardware evidence or a concrete need.

## Recent record

- 2026-07-29: Added and verified `sim.run_world`, which manages the complete
  PX4/Gazebo synthetic run with raw prompt boot detection, early-exit handling,
  process-group cleanup, motion/fault/recovery assertions, final CM5 zero,
  landing, and disarm. No simulator processes remained afterward.
- 2026-07-29: Extended the same runner with `--image` for the production
  RTP/YOLO/Kalman full-stack scenario. Both managed modes passed through PX4,
  CM5 safety, landing, disarm, and process cleanup.
- 2026-07-29: Strengthened the production full-stack verifier to require the
  voice-derived hover intent to reach the CM5 boundary during flight, not only
  the final shutdown zero. The managed production scenario passed again.
- 2026-07-29: Replaced the production verifier's fixed offboard-start sleep
  with an observed CM5 priming-setpoint condition, fixing an intermittent PX4
  `NO_SETPOINT_SET` race. Managed production SITL passed with the source fix.
- 2026-07-29: Made production verification logs explicitly report CM5 priming
  and in-flight visual-following/hover milestones; the managed run passed with
  those markers, obstacle backoff, landing, and disarm.
- 2026-07-29: Production verification now asserts the observed CM5 command
  stream ends with its shutdown zero. The managed RTP/YOLO/Kalman SITL run
  passed with priming, following, hover, obstacle backoff, shutdown zero,
  landing, and disarm.
- 2026-07-29: Shared flight preparation now attempts a landing if arming or
  takeoff setup fails. Production priming evidence is recorded only after the
  MAVSDK forwarder accepts the setpoint, fixing a real `NO_SETPOINT_SET` race;
  synthetic and production managed SITL both passed afterward.
- 2026-07-29: Hardened managed-run cleanup to terminate the dedicated PX4
  process group even when `make` exits before the scenario starts, avoiding
  broad name-based kills and partial-boot leftovers.
- 2026-07-28: Completed the simplification and safety pass: CM5 and Mac timing
  boundaries reject invalid numeric configuration; synthetic SITL requires
  actual forward, lateral, and obstacle-backoff telemetry; failures attempt
  landing; shared lifecycle verifies disarm; and the focused suite has 36
  checks. Production RTP/YOLO/Kalman SITL and UDP loopback pass.
- 2026-07-29: Consolidated the synthetic-world timeline into named phase
  constants shared by generation and assertions, removing repeated magic
  windows. The managed world run passed with complete telemetry and cleanup.
- 2026-07-28: Chose a deterministic synthetic behavioral world over a custom
  companion-owned Gazebo SDF. PX4's named worlds require external PX4 rebuild
  coupling and would not improve current control evidence; real camera, TOF,
  ROS, Wi-Fi, and first-flight behavior remain hardware gates.
