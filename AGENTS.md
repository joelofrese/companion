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

- The Mac brain uses images, dialogue, telemetry, and memory to choose intent
  and cautious movement.
- Local mode runs a subconscious VLM and conscious LLM in separate sessions.
- Gemini ER 2 Streaming is the simulation candidate for one persistent Mac
  brain.
- The visual model describes images and suggests cautious movement.
- When forward is blocked, the VLM may suggest a visible lateral alternate;
  Mac uses it only after the TOF stop clears.
- The LLM uses observations, dialogue, telemetry, and memory to choose intent.
- A follow goal passes its subject to the VLM as visual focus.
- Mac control turns that intent and visual suggestion into slow forward or
  lateral body-frame velocity; flight lifecycle owns altitude.
- A real intent change invalidates old visual context and pending brain
  results; rewording the same goal does not.
- A recognized dialogue intent stays active until a new open-ended request.
- When the LLM leaves a recognized request unanswered, Mac gives a short
  acknowledgement, then reports the first confirmed focused answer.
- A confirmed one-shot visual request holds motion until it is reported.
- Negative movement requests become hover before model interpretation.
- Low-confidence, stale, malformed, or missing input becomes zero motion.
- CM5 returns fresh TOF and vehicle telemetry, rejects unsafe commands,
  protects against obstacles, and is the final vehicle-side authority.
- PX4 stabilizes the vehicle and controls the motors.

The brain sends forward or lateral body-frame velocity only: never motor,
attitude, altitude, or absolute-position commands. A fresh obstacle reading
may override normal intent. Keep movement slow, deliberate, and easy to stop.

## Hardware boundary

Heavy perception, cognition, and interaction run on the Mac. The CM5 relays
video and telemetry, performs the final safety checks, and forwards approved
body-frame velocity setpoints to PX4, converting them with fresh vehicle
heading. Keep hardware-specific code on the CM5 so Mac behavior stays easy to
simulate.

The Ollama client handles local-model transport; `control/ollama_brain.py`
handles its prompts and response cleanup. `control/gemini_brain.py` keeps the
Gemini session and its small movement, hover, and speech tools on the Mac.

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
.venv/bin/python -m sim.run_world --explore --depth --gemini --trace --world objects --moving-person
.venv/bin/python -m sim.run_world --explore --camera --world objects --trace
.venv/bin/python -m sim.run_world --explore --camera --ollama --trace --world objects --request "look for the red box" --duration 20
.venv/bin/python -m sim.run_world --explore --faults --world default --duration 32
.venv/bin/python -m sim.run_world --explore --depth --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --depth --ollama --moving-person --world objects --intent "follow the person" --duration 20
.venv/bin/python -m sim.run_world --explore --duration 120
.venv/bin/python -m sim.run_world --image /Users/joelofrese/Code/Croppie/PX4-Autopilot/docs/assets/hardware/BeagleBone_Blue_balloons.jpg
.venv/bin/python -m sim.run_world --image /Users/joelofrese/Code/Croppie/PX4-Autopilot/docs/assets/hardware/BeagleBone_Blue_balloons.jpg --expect-person
```

`sim.run_world` manages PX4/Gazebo and cleanup. The deterministic synthetic
world checks motion, target loss, obstacles, visual detour recovery, invalid
and stale sensors, command faults, recovery, hover, shutdown, landing, and
disarm. The RTP image
scenario checks decoded video, deterministic person/non-person fixtures, Mac
commands, CM5 safety, PX4, landing, and disarm. `--expect-person` selects the
person fixture; without it the fixture stays stopped. The image still travels
through the complete RTP path. Real visual perception is checked through the
Gazebo camera with a brain model, not this deterministic fixture.
The deterministic timing, fault, and brain fixtures live in
`sim/world_fixture.py`; `sim/world.py` owns the PX4/Gazebo lifecycle and
verification.

`--camera` uses Gazebo's rendered camera as brain input. Without a model, it
checks camera transport through a zero-confidence placeholder and keeps motion
stopped. `--ollama` uses local VLM and LLM sessions; `--gemini` uses one
streaming Gemini session. Camera-only runs still stop because they have no TOF
reading. Use `--depth` when the brain should be allowed to move.
Use `--trace` to print visual observations, conscious decisions, model
latencies, and command reasons. Use `--snapshot PATH` to save the first
rendered frame for visual inspection. Use `--world`, `--duration`, `--request`,
`--intent`, and `--memory` to vary the world, run length, dialogue, goal, and
persistent experience. Typed dialogue also works during an exploratory run.
Add `--faults` to inject the normal obstacle,
sensor, link, invalid-command, and brain-shutdown schedule into an exploratory
run. Add `--headless` for unattended runs without the Gazebo GUI.

The companion-owned `objects` world adds simple colored shapes and a primitive
mannequin for visual exploration. The runner starts the vehicle at zero yaw
and leaves the Gazebo camera user-controlled. Camera and depth explorations
default to this world; other exploratory runs use the empty stock world.
Oversized simulation frames are reduced to the real 640-pixel camera width
before the VLM sees them. Because `objects` contains collidable objects, use
`--depth` for moving goals such as following. Camera-only runs have no forward
range reading, so they check visual behavior and bounded flight, not obstacle
clearance. Every non-default exploratory world requires `--camera` or
`--depth`; this prevents blind motion in collidable worlds.

`--depth` uses PX4's stock `x500_depth` model. Its RGB frames feed the Mac and
its depth readings feed CM5 safety. This is only a simulation approximation of
DEXI 3's forward TOF sensor, not a claim that DEXI 3 has a depth camera. Use
`--intent following` and `--pose x,y,z,roll,pitch,yaw` to start near an
obstacle.
Add `--moving-person` in the `objects` world to move its visible mannequin
between fixed poses through Gazebo's native pose service. This is a visual
interaction fixture, not a DEXI 3 hardware claim.
When Gazebo depth reaches the obstacle limit, the run also requires observed
CM5 backoff; runs that never reach it remain exploratory and only check bounded
behavior.

After installing Ollama and pulling local models:

```sh
ollama pull moondream
ollama pull gemma3:4b
.venv/bin/python -m control.companion <cm5-ip>
.venv/bin/python -m control.companion <cm5-ip> --dialogue
.venv/bin/python -m control.companion <cm5-ip> --ollama
```

Production defaults to one Gemini Robotics ER 2 Streaming session. It starts
with `explore the surroundings` unless `--intent` supplies another goal. Use
`--ollama` for separate local VLM and LLM sessions, defaulting to the faster
`moondream` for both. Set `--vlm-model` or `--llm-model` to use another local
model. Use `--dialogue` for typed conversation, `--voice-once` for one spoken
request, and `--memory` for editable experience memory.

With `GEMINI_API_KEY`, the optional Mac runner uses one persistent Gemini
Robotics ER 2 Streaming session. Exploratory simulation uses `--gemini`. It
receives the newest 640-pixel JPEG once per second while model turns run. A
high-level heartbeat follows each completed turn, never faster than once per
second. It can only request a brief forward or lateral move, hover, or speech.

## Current state

- The deterministic PX4/Gazebo mission and local UDP loopback verify the full
  command path, faults, recovery, safety, landing, and disarm.
- Exploratory camera and depth worlds exercise open-ended goals, dialogue,
  memory, bounded motion, and simulated TOF safety. Camera-only motion stops.
- Gemini ER 2 Streaming is the current Mac production path and the persistent
  brain for simulation. Local Ollama VLM and LLM sessions remain an explicit
  fallback. Live depth-world runs verified a focused red-box response, person
  following, reloadable experience memory, continuous one-frame-per-second
  video during model turns, bounded motion, the fault schedule, and a
  two-minute run with safety, landing, and disarm.
- The Mac sends only slow forward or lateral velocity. CM5 limits commands and
  uses TOF safety; PX4 stabilizes, lands, and disarms.
- Rendered Gazebo video, RTP video, simulated depth, and ROS forwarding exist;
  hardware remains unverified. DEXI 3 has no lidar.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not keep a long historical log.
