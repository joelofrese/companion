# Companion Drone

This is a short, living guide. Keep it accurate, simple, and editable.

## Goal

Build an autonomous indoor companion drone that notices, decides, moves
deliberately, and stays safe. Keep developing its capabilities so it can learn
over time.

Develop and test it in simulation whenever possible, so progress does not
depend on hardware.

## Priorities

1. Readable code and design.
2. Simple code and design.
3. Minimal code and design.

This makes the system easy for anyone to understand, debug, develop, and
maintain.

Remove dead code, speculative configuration, and abstractions without present
value. Refactor broadly when it makes the whole system simpler. Keep reviewing,
simplifying, and developing aligned companion capabilities without waiting for
confirmation.

## Git

Work directly on `main` in this single-contributor repository. Push verified
checkpoints there. Use a temporary branch only for risky isolated work, then
merge and delete it.

## Control flow

- The Mac runs a subconscious VLM and a conscious LLM in separate sessions.
- The VLM describes images and suggests cautious movement.
- The LLM uses observations, dialogue, telemetry, and memory to choose intent.
- Mac control turns that intent and visual suggestion into slow velocity.
- A real intent change invalidates old visual context and pending brain
  results; rewording the same goal does not.
- A recognized dialogue intent stays active until a new open-ended request.
- Low-confidence, stale, malformed, or missing input becomes zero motion.
- CM5 returns fresh TOF and vehicle telemetry, rejects unsafe commands,
  protects against obstacles, and is the final vehicle-side authority.
- PX4 stabilizes the vehicle and controls the motors.

The brain sends velocity only: never motor, attitude, or absolute-position
commands. A fresh obstacle reading may override normal intent. Keep movement
slow, deliberate, and easy to stop.

## Hardware boundary

Heavy perception, cognition, and interaction run on the Mac. The CM5 relays
video and telemetry, performs the final safety checks, and forwards approved
velocity setpoints to PX4. Keep hardware-specific code on the CM5 so Mac
behavior stays easy to simulate.

The target is the DroneBlocks DEXI 3: PX4, optical flow, a TOF distance sensor,
a Raspberry Pi camera, and a Raspberry Pi CM5. It has no lidar. Keep
simulation-only sensors separate from this hardware boundary.

## Simulation and validation

PX4 SITL with Gazebo is the primary development environment and the authority
for software flight behavior. Exercise the full control path, perception,
varied worlds, faults, recovery, safety, long runs, landing, and disarm.
Verify actual output or telemetry, including connection, readiness, arming,
setpoints, motion, safety intervention, landing, and disarm. Readiness uses
local position and magnetometer health, not global position or home health, so
the flight path does not require GPS.

Do not add unit tests. Prefer small end-to-end checks and real simulator
behavior so the code stays simple and flexible.

Keep two simulation modes:

- Deterministic missions prove flight, perception fixtures, transport, and
  safety behavior.
- Exploratory worlds give the brain an open-ended goal and let it choose what
  happens. Verify bounded motion, safety, landing, and disarm rather than exact
  decisions.

From `companion/`:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice
.venv/bin/python -m sim.command_loopback
.venv/bin/python -m sim.run_world
.venv/bin/python -m sim.run_world --explore --camera --ollama --trace --world walls
.venv/bin/python -m sim.run_world --explore --camera --world objects --trace
.venv/bin/python -m sim.run_world --explore --camera --ollama --trace --world objects --request "look for the red box" --duration 20
.venv/bin/python -m sim.run_world --explore --faults --camera --ollama --trace --world objects --duration 20
.venv/bin/python -m sim.run_world --explore --depth --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --duration 120
.venv/bin/python -m sim.run_world --image /Users/joelofrese/Code/Croppie/PX4-Autopilot/docs/assets/hardware/BeagleBone_Blue_balloons.jpg
.venv/bin/python -m sim.run_world --image /Users/joelofrese/Code/Croppie/PX4-Autopilot/docs/assets/hardware/BeagleBone_Blue_balloons.jpg --expect-person
```

`sim.run_world` manages PX4/Gazebo and cleanup. The deterministic synthetic
world checks motion, target loss, obstacles, invalid and stale sensors,
command faults, recovery, hover, shutdown, landing, and disarm. The RTP image
scenario checks decoded video, deterministic person/non-person fixtures, Mac
commands, CM5 safety, PX4, landing, and disarm. `--expect-person` selects the
person fixture; without it the fixture stays stopped. The image still travels
through the complete RTP path. Real visual perception is checked through the
Gazebo camera with Ollama, not this deterministic fixture.

`--camera` uses Gazebo's rendered camera as VLM input. Without `--ollama`, it
checks camera transport and keeps the brain stopped. With `--ollama`, local
VLM and LLM sessions make the run exploratory. Use `--trace` to print visual
observations, conscious decisions, and command reasons. Use `--world`,
`--duration`, `--request`, `--intent`, and `--memory` to vary the world, run
length, dialogue, goal, and persistent experience. Typed dialogue also works
during an exploratory run. Add `--faults` to inject the normal obstacle,
sensor, link, invalid-command, and brain-shutdown schedule into an exploratory
run.

The companion-owned `objects` world adds simple colored shapes and a primitive
mannequin for visual exploration. The runner starts the vehicle at zero yaw
and leaves the Gazebo camera user-controlled.
Oversized simulation frames are reduced to the real 640-pixel camera width
before the VLM sees them.

`--depth` uses PX4's stock `x500_depth` model. Its RGB frames feed the Mac and
its depth readings feed CM5 safety. This is only a simulation approximation of
DEXI 3's forward TOF sensor, not a claim that DEXI 3 has a depth camera. Use
`--intent following` and `--pose x,y,z,roll,pitch,yaw` to start near an
obstacle.

After installing Ollama and pulling local models:

```sh
ollama pull moondream
ollama pull gemma3:4b
.venv/bin/python -m control.companion <cm5-ip>
.venv/bin/python -m control.companion <cm5-ip> --dialogue
```

Production uses separate Ollama VLM and LLM sessions, defaulting to the faster
`moondream` and `gemma3:4b`. It starts with `explore the surroundings` unless
`--intent` supplies another goal. Set `--vlm-model` or `--llm-model` to change
the models; `qwen3-vl:2b` is an explicit slower visual option. Use
`--dialogue` for typed conversation, `--voice-once` for one spoken request,
and `--memory` for editable experience memory.

## Current state

- Deterministic Gazebo missions verify the full control path, perception
  fixtures, faults, recovery, brain failures, safety, hover, landing, and
  disarm.
- Exploratory camera and depth worlds run local VLM/LLM brains with dialogue,
  visual focus, memory, bounded motion, simulated TOF safety, landing, and
  disarm.
- RTP person/no-person fixtures verify the video, Mac, CM5, and PX4 path;
  rendered Gazebo frames verify the real local VLM path.
- Focused Qwen runs identify simple objects and make bounded movement;
  faster Moondream remains conservative when the scene is unclear.
- Exploratory `--faults` runs reuse the deterministic safety schedule without
  requiring exact brain decisions, while checking the observed CM5 safe
  outputs for each reached fault.
- Visual or vehicle telemetry that is stale, missing, malformed, or unsafe
  stops Mac motion. CM5 remains the final authority.
- Intent changes invalidate old visual context; same-goal rewording does not.
  Experience memory persists across exploratory runs.
- Conscious thoughts keep the active visual focus across new observations and
  ignore generic model placeholders.
- Simulations start at zero yaw and keep that heading through offboard
  handoff. They keep the camera user-controlled and resize frames to the real
  640-pixel camera width. The GUI opens after PX4 spawns the model so it does
  not reframe during takeoff. Depth is only a TOF approximation; DEXI 3 has no
  lidar.
- The ROS 2 velocity seam exists. DEXI 3 hardware and optical-flow quality
  remain unverified in SITL.
- Keep prioritizing closed-loop autonomous world operation and simpler code.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not keep a long historical log.
