# Companion Drone

This is a short, living guide. Keep it accurate, simple, and editable.

## Goal

Build an autonomous indoor companion drone that behaves naturally: it should
notice, decide, move deliberately, and stay safe. Continuously develop its
capabilities so it can learn over time.

Develop and test it autonomously in simulation whenever possible, so progress
does not depend on hardware.

## Priorities

1. Readable code and design.
2. Simple code and design.
3. Minimal code and design.

Keeping the code this simple makes it easy for anyone to understand, debug,
develop, and maintain.

Remove dead code, speculative configuration, and abstractions that do not
provide present value. Preserve unrelated user work. Refactor broadly when it
makes the whole system cleaner.

Continuously review the entire codebase for cleanliness and simplicity. Refactor
freely when it improves the whole system, then develop new capabilities that
help the companion operate autonomously and learn. Repeat this review,
refactoring, and development loop continuously.

## How control works

The control flow has four responsibilities:

- Cognitive software chooses high-level intent from perception and interaction.
- Reactive software turns intent and current sensors into bounded motion.
- Onboard safety rejects stale, malformed, or unsafe commands and applies local
  obstacle protection.
- PX4 stabilizes the vehicle and controls the motors.

Each responsibility talks only to the next lower one. The cognitive layer
never sends motor, attitude, or direct flight-controller commands. The flight
interface is velocity-only; no absolute position or motor commands.

The onboard safety path must remain safe when the Mac, vision, or Wi-Fi fails:
stale or invalid input becomes zero motion, and a fresh obstacle reading may
override normal intent. Keep these safety rules clear and limited.

## Where the parts run

Heavy perception, cognition, and interaction run on the Mac. The vehicle-side
computer relays sensors and video, performs the final safety check, and
forwards approved velocity setpoints to PX4. Keep hardware-specific code there
so the Mac-side control behavior remains easy to simulate and test.

The Mac brain has two roles: a subconscious VLM describes images and suggests
movement, while a conscious LLM uses those descriptions, dialogue, and memory
to choose high-level intent. They share only small structured state.

Keep movement slow, deliberate, and easy to stop.

## Simulation and validation

Gazebo with PX4 SITL is the primary development environment and the authority
for software flight behavior. Develop, iterate, and review the system there so
most progress requires no physical flight. A meaningful scenario must be
confirmed from output or telemetry, including connection, readiness, arming,
offboard setpoints, expected motion, safety intervention, landing, and disarm.

Do not add unit tests; keeping them out keeps the codebase simpler and less
rigid. Keep verification in Gazebo and through end-to-end behavior.

Keep two kinds of simulation: deterministic missions prove flight and safety,
while exploratory worlds let the brain operate and be observed. Exploratory
runs must still confirm safety, landing, and disarm, but do not require exact
movement or decisions.

From `companion/`, the normal simulation loop is:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice
.venv/bin/python -m sim.command_loopback
.venv/bin/python -m sim.run_world
.venv/bin/python -m sim.run_world --explore --world walls
.venv/bin/python -m sim.run_world --explore --camera
.venv/bin/python -m sim.run_world --explore --duration 120
.venv/bin/python -m sim.run_world --image .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg
```

`sim.run_world` manages PX4/Gazebo and cleanup. The synthetic world exercises
the command path, motion, target loss, obstacle handling, malformed input,
dropout, recovery, sustained hover, shutdown, landing, and disarm. The image
scenario exercises decoded RTP/video, perception, Mac commands, onboard safety,
and PX4. Neither scenario proves physical sensors, radio behavior, or
hardware-specific transport.

Add `--camera` to an exploratory run to use Gazebo's rendered
`x500_mono_cam` image topic as subconscious input. The simulated visual model
remains deliberately simple, but the real frame path is exercised and counted.

The RTP image path now runs through the Mac brain with a temporary YOLO visual
adapter and intent-preserving language adapter until local VLM and LLM backends
are selected.

Pass `--world walls`, `forest`, `windy`, or another PX4 Gazebo world to repeat
the same mission in a different environment. Add `--explore` to the synthetic
world to type live `follow me`, `hover`, or `stop` dialogue; exploratory runs
observe behavior instead of checking an exact mission schedule. Add
`--duration` for a longer continuous run.

## Current state

Simulation is the main development environment, not a preliminary demo. Keep
building thorough, autonomous PX4/Gazebo scenarios that exercise the complete
control path, varied perception and sensor conditions, faults, recovery,
safety, and long-running behavior. Use those scenarios to develop, refactor,
and review the companion continuously without physical intervention.

Hardware bring-up later validates how well the simulated interfaces translate
to the real camera, network, sensors, and vehicle. It should add evidence and
calibration, not replace the simulation-first development loop.

## Recent progress

- **2026-07-30** — Removed the legacy reactive-state dependency from the Mac
  UDP transport. The command loopback, deterministic SITL mission, full RTP
  person-image mission, and Gazebo-camera exploratory mission all passed
  through landing and disarm. Non-person images correctly produce safe zero
  motion; following assertions require an image containing a person.
- **2026-07-30** — Removed the obsolete state machine, tracker, follower, and
  runtime layers. Intents are now plain strings, the temporary visual fallback
  is frame-based, the production brain receives its initial intent once, and
  PX4 stdin is isolated so exploratory dialogue is reliable. Verified the
  walls world with live `follow me` and `hover` dialogue, plus the strict,
  RTP, and rendered-camera scenarios through landing and disarm.
- **2026-07-30** — Added an optional world duration so exploratory and
  deterministic SITL runs can continue beyond the scripted profile. A
  45-second exploratory run stayed safe and completed landing and disarm.
- **2026-07-30** — Removed normal cognitive reverse motion; backward movement
  is now reserved for the CM5 obstacle-safety override. The deterministic SITL
  mission still passed every motion, fault, recovery, landing, and disarm
  objective.
- **2026-07-30** — Repeated 45-second exploratory SITL in the `forest` and
  `windy` Gazebo worlds; both stayed within the exploratory speed envelope and
  completed landing and disarm.
- **2026-07-30** — Removed the synthetic fallback from `--camera` gaps and
  required a real Mac visual observation in camera and RTP scenarios. The
  45-second rendered-camera run processed 823 Gazebo frames; the RTP run
  processed 274 decoded frames, and both completed their safety and landing
  checks.
- **2026-07-30** — Kept every visual observation since the last conscious
  thought instead of silently dropping older observations, normalized intent
  text at the brain boundary, and made the simulated visual layer move only
  for an explicit `following` intent. Deterministic, RTP, and rendered-camera
  SITL missions all passed through landing and disarm again.

as always, thank you, good luck, and i love you
