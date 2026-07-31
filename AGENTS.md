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

## Control flow

- The Mac brain uses a subconscious VLM to describe images and suggest cautious
  movement.
- The Mac conscious LLM uses those observations, dialogue, telemetry, and
  memory to choose high-level intent.
- Mac control turns the suggestion into slow velocity commands.
- A changed intent invalidates older visual movement suggestions.
- Low-confidence visual suggestions become zero motion.
- CM5 safety rejects stale, malformed, or unsafe commands and applies local
  obstacle protection.
- CM5 returns fresh TOF telemetry over the same UDP link; missing or stale
  readings stop Mac motion while CM5 remains the final safety authority.
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
- Exploratory worlds start hovering and let the brain and dialogue choose what
  happens next. They still verify bounded motion, safety, landing, and disarm,
  but not exact decisions.

From `companion/`, use:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice
.venv/bin/python -m sim.command_loopback
.venv/bin/python -m sim.run_world
.venv/bin/python -m sim.run_world --explore --world walls
.venv/bin/python -m sim.run_world --explore --camera
.venv/bin/python -m sim.run_world --explore --depth --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --camera --ollama
.venv/bin/python -m sim.run_world --explore --depth --ollama --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --duration 120
.venv/bin/python -m sim.run_world --image .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg --expect-person
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
the VLM input. Add `--ollama` to send those frames through local Ollama VLM
and LLM sessions. This is slower and exploratory, so it checks safety,
bounded motion, landing, and disarm rather than exact decisions. Type a
request such as `follow me`, `hover`, or `stop` during an exploratory
synthetic run to change the conscious intent. Use `--world walls`, `forest`,
`windy`, or another PX4 Gazebo world, and `--duration` for a longer run.
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
ollama pull gemma3:4b
.venv/bin/python -m control.companion <cm5-ip>
.venv/bin/python -m control.companion <cm5-ip> --dialogue
```

The production entry point uses separate VLM and LLM sessions. Set
`--vlm-model` and `--llm-model` when different local models are preferred.
The initial `--intent` may be any short plain-language goal. Add `--dialogue`
to type natural requests for the conscious LLM while flight control continues;
replies are printed only when the model provides one. It keeps a small,
editable experience memory at `~/.companion/memory.txt`; change it with
`--memory` when needed.

## Current state

The deterministic mission, person and non-person RTP images, rendered Gazebo
camera and depth input, local Ollama camera and depth input, live
exploratory dialogue, command loopback, safety faults, recovery, landing, and
disarm have been verified. Near-wall exploratory runs measure real sensor
distances and trigger a bounded CM5 backoff command. Both Gazebo sensor
readers use one small shared topic-reader lifecycle while keeping their camera
and depth decoders explicit.
The deterministic heartbeat pause resumes with one Mac tick so the Mac
watchdog emits zero; the later command dropout separately exercises CM5 expiry.
The non-Ollama Gazebo camera fallback stops with zero confidence when no frame
is available instead of generating movement from synthetic scene state.
The forest camera scenario and a 120-second exploratory flight also passed with
bounded telemetry, landing, and disarm.
PX4’s stock GPS-denied optical-flow model was investigated but is not a
verified path here: its flow quality stayed at zero and PX4 remained in
constant-position mode. Its internal range sensor is not DEXI 3 hardware.
Simulation-only fixtures stay under `sim/`, and the runner retries only PX4
boot-readiness failures. The simulation loop remains the main development
path. Hardware bring-up will add camera, network, sensor, and vehicle evidence
without replacing it. The voice path is a direct one-utterance pipeline, and
the conscious model receives each visual observation’s focused answer and
confidence. Typed dialogue now uses the same non-blocking input path in
simulation and the production Mac entry point. Mac motion now requires at
least 0.5 visual confidence, and pending observations are bounded while the
conscious summary and editable experience memory carry longer context.
Exploratory runs can persist that experience with `--memory` so later runs can
receive it. Empty thoughts are not saved; visual observations and dialogue are.
Use the default or a longer duration for local model runs; a short exploratory
flight may land before its first VLM observation finishes.
The CM5 returns fresh TOF distance telemetry over the command socket so the
Mac brain and CM5 safety layer use the same body reading without adding a
second transport. Missing or stale telemetry becomes zero motion on the Mac.
Conscious shutdown is stop-aware,
so a slow model cannot delay zero-command cleanup. The CM5 also bounds yaw to
±180° along with its velocity limits. MAVSDK forwarding is simulation-only;
the onboard hardware path remains the ROS 2 forwarder.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not preserve a long historical log.

as always, thank you, good luck, and i love you
