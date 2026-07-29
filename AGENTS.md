# Companion Drone

This is a living guide. Keep it short, current, and editable. At the end of a
session, record only decisions or evidence that future work needs; delete stale
history rather than growing a changelog.

## Purpose

Build a slow, natural indoor companion drone. The AI chooses behavior; lower
layers turn that behavior into safe motion. It should feel creature-like, not
like a point-to-point robot.

## Working rules

- Prefer the simplest complete design. Readability, small interfaces, and clear
  control flow outrank feature speed.
- Remove obsolete code, tests, comments, wrappers, and compatibility paths.
  Add code, files, configuration, dependencies, and abstractions only when they
  solve a present problem.
- Work autonomously. At each meaningful milestone, read `STEERING.md` once
  before selecting the next work; do not poll it during implementation.
- Make the smallest complete vertical slice, then simplify the final diff.
  Preserve a working baseline before substantial restructuring.
- Preserve unrelated user changes. Work on a feature branch; make verified,
  cohesive, reversible Git checkpoints and push stable ones when possible.
- If hardware or credentials block one task, record the boundary and continue
  with useful simulation or code work. Never weaken safety to make progress.

## Safety architecture

The layers communicate only with the layer immediately below:

| Layer | Runs on | Responsibility |
|---|---|---|
| Cognitive | Mac | LLM, vision, voice, and slow state transitions |
| Reactive | Mac | Converts state and perception into bounded velocity commands |
| Stabilization | PX4 | Attitude and motor control |

The Mac sends only velocity intent. The CM5 is an independent final safety
boundary: it rejects malformed, stale, reordered, or out-of-bounds UDP commands
and lets a fresh forward obstacle reading override them. PX4 receives only
velocity setpoints; no layer sends motors or absolute positions.

The state model is `IDLE → FOLLOWING → AVOIDING → HOVERING → RESPONDING → IDLE`.
Cognitive intent owns the persistent state. Obstacle safety can temporarily
report `AVOIDING` and command bounded backoff, then restore the saved intent.
Normal visual following closes distance or holds; only obstacle safety reverses.

## System boundaries

- Drone: DroneBlox DEXI 3 with PX4 (FMUv6X), Raspberry Pi CM5, optical flow,
  forward TOF, and Pi camera.
- Mac: all ML and high-level control. CM5: sensor/video relay and final command
  safety/forwarding. Link: local Wi-Fi.
- Follow geometry is deliberately conservative: tracked target height controls
  north correction and horizontal image error controls east correction. Kalman
  prediction compensates for vision delay.
- Push-to-talk Whisper is the initial voice interface. Unknown or conflicting
  non-safety speech does nothing; explicit stop/hover is safe.
- Use DEXI-OS rather than a custom Pi image. Do not modify power electronics or
  add a tether. Hardware flight is brief and follows simulation evidence.

## Validation

- Tests protect distinct observable behavior, safety properties, or active
  integration contracts. Delete redundant, implementation-coupled tests.
- Use bounds and tolerances for tunable flight behavior. Unit tests are fast
  evidence; PX4 SITL/Gazebo is the authority for flight behavior.
- Confirm a simulator run from its output and telemetry, not process survival.
  A full-stack pass must show connection, readiness, arm, offboard motion,
  obstacle backoff, landing, and disarm.
- Fix failures at their source. Do not skip checks, weaken assertions, or add
  retries merely to pass.

Useful checks from `companion/`:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m unittest discover -s tests -q
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice tests
python -m sim.command_loopback
python -m sim.world
python -m sim.offboard_full .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg
```

Start PX4 SITL in `~/Code/Croppie/PX4-Autopilot` with
`stdbuf -oL -eL make px4_sitl gz_x500`; wait for `pxh>` before a verifier. If a
verifier hangs at connection, check PX4 has UDP port 14580; a flaky boot can
omit that link. Clean orphaned PX4/Gazebo/MAVSDK processes before retrying.

## Hardware operation

- Mac production service: `python -m control.companion <cm5-ip>`. It defaults
  to IDLE; use `--state following` or one `--voice-once` utterance deliberately.
- CM5 camera sender: `python -m onboard.video_sender <mac-ip>`.
- CM5 ROS bridge: `python -m onboard.ros2_bridge`. It owns PX4 velocity
  setpoints and must not run beside `px4_offboard_manager`, which emits
  position-mode commands.
- Before DEXI-OS bring-up, run `python -m onboard.bringup_check`. ROS 2,
  `px4_msgs`, Micro-ROS serial, `libcamerasrc`, `/dev/ttyAMA2`, the physical
  distance topic, and Wi-Fi streaming remain hardware validation boundaries.

## Current state

- PX4 SITL/Gazebo, MAVSDK offboard velocity control, state/reactive control,
  YOLO/Kalman tracking, RTP/H.264 receiving, voice intent, UDP command safety,
  CM5 service/ROS bridge seams, and full-stack software SITL are implemented.
- The active Mac path is one `CompanionRuntime.tick`: vision → intent → reactive
  command → watchdog → UDP. The CM5 path independently validates and forwards
  only safe commands.
- `sim.world` is the autonomous behavioral harness: synthetic-world target truth
  derives image geometry from PX4's simulated NED position, then drives the real
  Mac, CM5, PX4, and Gazebo path through follow, lateral motion, target loss,
  obstacle, command-dropout, and invalid-command scenarios. It is deliberately
  not a claim of camera or TOF realism.
- Hardware remains pending: real CM5 Wi-Fi video, DEXI ROS serial/PX4 response,
  TOF topic/wiring, and the first flight.
- Open design question: one forward TOF sensor supports stopping/backoff but not
  full directional avoidance; do not invent additional sensing until hardware
  evidence or a concrete requirement calls for it.

## Recent record

- 2026-07-28: Added `python -m sim.world`, a deterministic PX4/Gazebo behavior
  scenario. It passed forward following, lateral target tracking, target-loss
  hold, obstacle backoff, command-dropout expiry, invalid-command rejection,
  and clean landing through the production Mac→CM5 safety path. Target geometry
  now responds to the vehicle's actual simulated NED position, and the scenario
  verifies saved following intent resumes after obstacle backoff. The verifier
  explicitly closes its MAVSDK helper for clean repeated runs.
- 2026-07-28: Tightened `sim.offboard_full` so its production RTP/YOLO scenario
  must observe both nonzero visual-following motion and obstacle backoff before
  it can pass. The bundled YOLO person fixture produced 0.04 m/s forward,
  -0.17 m/s backoff, and clean landing; non-person images are correctly rejected
  as visual-following evidence.
- 2026-07-28: Reduced deterministic coverage to 32 focused safety-contract
  checks across command validity, watchdog/obstacle fail-safe behavior, safe
  shutdown, vision liveness, and CM5/PX4 bridge seams. Removed component and
  adapter-detail tests. The full PX4/Gazebo scenario still passed with -0.18
  m/s obstacle backoff and a clean landing.
- 2026-07-28: Simplified the Mac control path by removing separate step and loop
  wrappers. Removed redundant video/YOLO simulation wrappers and eight
  wrapper/mock-only tests. The retained deterministic suite passed 120 tests;
  the full SITL run observed 0.15 m/s forward, -0.20 m/s obstacle backoff, and
  a clean landing.
- 2026-07-28: The previous full-stack SITL run validated production RTP/H.264,
  YOLO/Kalman, Mac UDP control, CM5 safety service, obstacle backoff, and clean
  landing. Hardware transport and sensor validation remain separate.
