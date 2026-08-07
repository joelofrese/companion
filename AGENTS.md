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
- A changed intent invalidates old visual context and pending brain results.
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
.venv/bin/python -m sim.run_world --explore --depth --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --duration 120
.venv/bin/python -m sim.run_world --image .venv/lib/python3.9/site-packages/ultralytics/assets/zidane.jpg --expect-person
.venv/bin/python -m sim.run_world --image /Users/joelofrese/Code/Croppie/PX4-Autopilot/docs/assets/hardware/BeagleBone_Blue_balloons.jpg
```

`sim.run_world` manages PX4/Gazebo and cleanup. The deterministic synthetic
world checks motion, target loss, obstacles, invalid and stale sensors,
command faults, recovery, hover, shutdown, landing, and disarm. The RTP image
scenario checks decoded video, perception fixtures, Mac commands, CM5 safety,
PX4, landing, and disarm. YOLO is only a repeatable image fixture, not the
production brain. A person image needs `--expect-person`; a non-person image
should remain stopped.

`--camera` uses Gazebo's rendered camera as VLM input. Without `--ollama`, it
checks camera transport and keeps the brain stopped. With `--ollama`, local
VLM and LLM sessions make the run exploratory. Use `--trace` to print visual
observations, conscious decisions, and command reasons. Use `--world`,
`--duration`, `--request`, `--intent`, and `--memory` to vary the world, run
length, dialogue, goal, and persistent experience. Typed dialogue also works
during an exploratory run.

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
  fixtures, faults, recovery, safety, hover, landing, and disarm.
- Simulations start at zero yaw and hold that heading through takeoff, so the
  model does not make a repeated visible turn.
- Exploratory camera and depth runs work in varied PX4 worlds with local
  brains, bounded motion, simulated TOF safety, landing, and disarm.
- A near-wall local-brain depth run observed 0.55 m and CM5 backoff at
  -0.20 m/s before landing and disarm.
- Fresh visual results permit movement only briefly; stale results, camera
  gaps, stale sensors, malformed commands, low confidence, command loss,
  obstacles, and model shutdown stop safely.
- Ollama VLM and LLM sessions support structured observations, intent,
  dialogue, focused vision, memory, and voice. Trace output shows their
  observable decisions and Mac/CM5 command reasons.
- Scripted hover and follow requests are verified through the rendered-camera
  path; hover stays still and follow remains bounded.
- Experience memory persists across exploratory runs and is available to the
  next conscious decision.
- The ROS 2 velocity seam exists. DEXI 3 hardware and optical-flow quality
  remain unverified in SITL.
- Keep prioritizing closed-loop autonomous world operation and simpler code.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not keep a long historical log.
