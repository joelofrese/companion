# Companion Drone

This is a short, living guide. Keep it accurate, simple, and editable.

## Goal

Build an autonomous indoor companion drone that notices, decides, moves
deliberately, and stays safe. Continuously develop its capabilities so it can
learn over time.

Develop and test it autonomously in simulation whenever possible, so progress
does not depend on hardware.

## Priorities

1. Readable code and design.
2. Simple code and design.
3. Minimal code and design.

Keeping the code this simple makes it easy for anyone to understand, debug,
develop, and maintain.

Remove dead code, speculative configuration, and abstractions without present
value. Refactor broadly when it makes the whole system cleaner. Continuously
review the codebase, simplify it, and then develop aligned companion
capabilities. Repeat that loop without waiting for confirmation.

## Git

Work directly on `main` in this single-contributor repository. Push verified
checkpoints there; use a temporary branch only for risky isolated work, then
merge and delete it.

## Control flow

- The Mac brain uses a subconscious VLM to describe images and suggest cautious
  movement.
- The Mac conscious LLM uses those observations, dialogue, telemetry, and
  memory to choose high-level intent.
- Mac control turns the suggestion into slow velocity commands.
- A changed intent invalidates older visual context and pending brain results.
- A recognized dialogue intent stays active until a new open-ended dialogue
  request releases it.
- Low-confidence visual suggestions become zero motion.
- CM5 safety rejects stale, malformed, or unsafe commands and applies local
  obstacle protection.
- CM5 returns fresh TOF and vehicle telemetry over the same UDP link; missing
  or stale readings stop Mac motion while CM5 remains the final safety
  authority.
- PX4 stabilizes the vehicle and controls the motors.

The brain never sends motor or attitude commands. The flight interface is
velocity-only: no absolute position or motor commands. Stale or invalid input
becomes zero motion, and a fresh obstacle reading may override normal intent.

## Where code runs

Heavy perception, cognition, and interaction run on the Mac. The CM5 relays
video and sensor telemetry, performs the final safety check, and forwards
approved velocity setpoints to PX4. Keep hardware-specific code on the CM5
side so Mac behavior stays easy to simulate.

The target hardware is the DroneBlocks DEXI 3: PX4, optical flow, a TOF
distance sensor, a Raspberry Pi camera, and a Raspberry Pi CM5. It has no
lidar. Keep simulation sensors clearly separate from this hardware boundary.

Keep the VLM and conscious LLM as separate sessions with small structured data
between them. Keep movement slow, deliberate, and easy to stop.

## Simulation and validation

PX4 SITL with Gazebo is the primary development environment and the authority
for software flight behavior. Build thorough scenarios that exercise the full
control path, perception, varied worlds, faults, recovery, safety, long runs,
landing, and disarm. Verify actual output or telemetry, including connection,
readiness, arming, offboard setpoints, motion, safety intervention, landing,
and disarm.
Simulation readiness checks local position and magnetometer health, not global
position or home health, so the flight path does not require GPS.

Do not add unit tests. They add rigidity without helping the project’s
simulation-first goal. Prefer small end-to-end checks and real simulator
behavior.

Keep two simulation modes:

- Deterministic missions prove flight, perception fixtures, transport, and
  safety behavior.
- Exploratory worlds start with an open-ended goal and let the brain and
  dialogue choose what happens next. They still verify bounded motion, safety,
  landing, and disarm, but not exact decisions.

From `companion/`, use:

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
world exercises motion, target loss, obstacle handling, invalid and stale
sensor data, command faults, recovery, hover, shutdown, landing, and disarm.
The RTP image scenario exercises decoded video, perception, Mac commands,
malformed and stale packets, CM5 safety, PX4, landing, and disarm. Add
`--expect-person` for a person fixture; omit it to verify that a non-person
image stays stopped. These scenarios do not prove physical sensors, radio, or
hardware-specific transport.

Add `--camera` to an exploratory run to use Gazebo’s rendered camera topic as
the VLM input. Without `--ollama`, the camera transport is verified but the
brain stays stopped because no visual model is configured. Add `--ollama` to
send those frames through local Ollama VLM and LLM sessions. This is slower
and exploratory, so it checks safety, bounded motion, landing, and disarm
rather than exact decisions. Type a request such as `follow me`, `hover`, or
`stop` during an exploratory synthetic run to change the conscious intent.
Use `--world walls`, `forest`, `windy`, or another PX4 Gazebo world, and
`--duration` for a longer run.
With `--ollama`, keep the default 32-second duration or use longer; local model
startup can make a shorter run land safely before its first conscious thought.
Use `--request` to inject one dialogue request without typing, which makes
the conscious interaction path repeatable in unattended simulation.
Use `--trace` to print each meaningful structured VLM observation, conscious decision,
and the reason the Mac or CM5 held or changed the command. It shows decisions,
not hidden model reasoning, and is useful when an exploratory run appears to
hover.
Use `--intent TEXT` to replace the default open-ended goal with any short
high-level goal; explicit stop/hover goals remain stationary, while a clear
follow goal drives the synthetic person fixture.
Add `--memory PATH` to persist conscious experience across exploratory runs.

Add `--depth` to use PX4’s stock `x500_depth` model. Its rendered RGB frames
feed the Mac brain and its depth readings feed CM5 safety. This is a
simulation-only approximation of the DEXI 3’s forward TOF distance sensor,
not a claim that DEXI 3 has a depth camera. Use `--intent following` and
`--pose x,y,z,roll,pitch,yaw` to start near a world obstacle and observe the
simulated sensor trigger local safety.

The deterministic RTP scenario uses YOLO only as a repeatable person-image
fixture. It is not the production brain. Production `control/` uses local
Ollama models; simulation-only fixtures live under `sim/`.

After installing Ollama and pulling a local vision-capable model:

```sh
ollama pull moondream
ollama pull gemma3:4b
.venv/bin/python -m control.companion <cm5-ip>
.venv/bin/python -m control.companion <cm5-ip> --dialogue
```

The production entry point uses separate VLM and LLM sessions. It defaults to
the faster `moondream` for vision and `gemma3:4b` for conscious language so the
two sessions can run independently; set `--vlm-model` or `--llm-model` when a
different local model is preferred. `qwen3-vl:2b` remains an explicit option
when broader visual reasoning is worth the added latency.
The initial `--intent` may be any short plain-language goal. Add `--dialogue`
to type natural requests for the conscious LLM while flight control continues;
replies are printed only when the model provides one. It keeps a small,
editable experience memory at `~/.companion/memory.txt`; change it with
`--memory` when needed.
Use `--voice-once` to route one spoken request through the same dialogue path;
explicit movement intents remain safety-checked, while open-ended requests
reach the conscious LLM.

## Current state

- Deterministic Gazebo missions verify the full control path, perception
  fixtures, faults, recovery, safety, hover, landing, and disarm.
- Velocity commands contain only NED velocities; PX4 holds its heading, and
  synthetic simulation checks actual heading telemetry for rotation.
- Exploratory camera and depth runs work across varied PX4 worlds with local
  brains, bounded motion, simulated TOF safety, landing, and disarm; the
  near-wall depth path has observed 0.56m and forwarded CM5 backoff.
- A 120-second moving-platform exploration completed 2,364 visual observations
  and 240 conscious cycles with heading hold, landing, and disarm.
- Ollama VLM and LLM sessions produce structured observations and intent; the
  conscious loop also supports dialogue, focused vision, memory, and voice.
- Trace output shows VLM observations, conscious decisions, and Mac/CM5 command
  reasons without exposing hidden model reasoning.
- Stale frames or sensors, malformed commands, low confidence, command loss,
  obstacles, and model shutdown fail safely; CM5 remains the final authority.
- The ROS 2 velocity seam exists, but DEXI 3 hardware and its optical-flow
  quality remain unverified in SITL.
- Continue prioritizing closed-loop autonomous world operation and simpler
  code over narrow new behaviors.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not preserve a long historical log.
