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
- A follow goal passes its subject to the VLM as visual focus.
- Mac control turns that intent and visual suggestion into slow body-frame
  velocity.
- A real intent change invalidates old visual context and pending brain
  results; rewording the same goal does not.
- A recognized dialogue intent stays active until a new open-ended request.
- Negative movement requests become hover before model interpretation.
- Low-confidence, stale, malformed, or missing input becomes zero motion.
- CM5 returns fresh TOF and vehicle telemetry, rejects unsafe commands,
  protects against obstacles, and is the final vehicle-side authority.
- PX4 stabilizes the vehicle and controls the motors.

The brain sends body-frame velocity only: never motor, attitude, or
absolute-position commands. A fresh obstacle reading may override normal
intent. Keep movement slow, deliberate, and easy to stop.

## Hardware boundary

Heavy perception, cognition, and interaction run on the Mac. The CM5 relays
video and telemetry, performs the final safety checks, and forwards approved
body-frame velocity setpoints to PX4, converting them with fresh vehicle
heading. Keep hardware-specific code on the CM5 so Mac behavior stays easy to
simulate.

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
.venv/bin/python -m sim.run_world --explore --faults --world default --duration 32
.venv/bin/python -m sim.run_world --explore --depth --world walls --intent following --pose 3.8,0,0,0,0,0
.venv/bin/python -m sim.run_world --explore --depth --moving-person --world objects --intent "follow the person" --duration 20
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
The deterministic timing, fault, and brain fixtures live in
`sim/world_fixture.py`; `sim/world.py` owns the PX4/Gazebo lifecycle and
verification.

`--camera` uses Gazebo's rendered camera as VLM input. Without `--ollama`, it
checks camera transport through a zero-confidence placeholder and keeps motion
stopped. With `--ollama`, local VLM and LLM sessions make the run exploratory,
but CM5 keeps motion stopped because this model has no TOF reading. Use
`--depth` when the brain should be allowed to move.
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
- Exploratory camera and depth worlds run the Mac VLM/LLM path with dialogue,
  visual focus, memory, bounded motion, and simulated TOF safety. Camera-only
  motion stays stopped because DEXI 3 has no forward range reading in that
  mode.
- The `objects` world provides simple colored objects and a mannequin in the
  forward camera view; `--moving-person` moves that mannequin between fixed
  poses on both sides and exercises changing visual scenes. The default
  moondream model now describes concrete broad scenes with the shorter visual
  prompt. A 20-second open-ended run produced six observations and two
  conscious thoughts, then issued bounded left and forward movement; a current
  focused red-box run produced four confirmed answers and retained focus while
  holding zero.
  A current 45-second moving-person follow run produced 13 person observations,
  repeated bounded forward suggestions, 0.24m/s maximum telemetry, and CM5
  backoff when the red box reached 0.49m; lateral position remains
  model-limited. The slower Qwen and Gemma vision options produced no thought
  in 20 seconds.
- Non-default worlds require rendered camera or simulated depth so motion is
  never blind in a collidable world. Gazebo frames are reduced to 640 pixels.
- Synthetic open-ended goals can move through clear simulated space. Real
  open-ended VLM runs may hold zero when the model chooses stop; focused follow
  can issue slow forward commands when the person is recognized. Dialogue can
  change intent or visual focus; natural requests such as "follow me" focus the
  person, and an intent such as "inspect the red box" can provide focus when
  the LLM leaves that field empty. Experience memory is editable, persists
  across runs, and records command, velocity, and obstacle outcomes;
  exploratory simulation reloads and verifies those records.
- If the conscious model omits a summary, Mac keeps the latest visual
  observation as context.
- Unanswered visual focus stays active; answered focus can be released or
  replaced by the conscious model.
- Stale, missing, malformed, low-confidence, or failed brain and sensor input
  stops Mac motion. CM5 still rejects unsafe or stale commands and remains the
  final authority.
- `--trace` shows observations, decisions, latencies, and command reasons;
  `--snapshot` saves a rendered frame for inspection; `--faults` checks the
  same safety schedule through synthetic sensors and live Gazebo depth without
  requiring exact brain decisions.
- A fault-injected live-depth run verified 584/584 valid depth samples, CM5
  backoff, stale and invalid depth, command dropout and rejection, missing
  velocity, brain shutdown, landing, and disarm; forward motion stayed bounded.
- Conscious prompts keep distinct visual changes and drop repeated descriptions
  so local thinking stays responsive.
- VLM prompt echoes and intent-only descriptions become unclear, zero-confidence
  observations rather than movement suggestions.
- Prompt-field echoes from local models are discarded at the Ollama boundary.
  An opt-in moondream conscious run produced seven thoughts in 20 seconds,
  honored follow and stop dialogue, and issued bounded follow motion; Gemma
  remains the default conscious model until broader dialogue shows equal
  reliability.
- Exploration now asks the VLM to move slowly on a clear path and stop below
  the shared obstacle limit. A live run reached 0.24m/s forward and -0.18m/s
  right velocity; a near-wall run reached 0.50m and observed CM5's -0.20m/s
  backoff before landing and disarm.
- Brain and CM5 use slow forward, right, and down body-frame velocity. Both
  simulated and ROS 2 CM5 paths convert it with fresh vehicle heading.
- A 120-second synthetic exploratory run completed with 2,331 VLM observations
  and 240 conscious thoughts, bounded motion, landing, and disarm. Near-wall
  depth runs also observe CM5 backoff. RTP fixtures and rendered Gazebo camera
  runs verify the video, Mac, CM5, and PX4 paths.
- The target remains the DEXI 3: no lidar, DEXI hardware is unverified, and
  current SITL's GPS-denied optical-flow topic publishes no messages, so no
  flow-dependent behavior is claimed.
- Keep prioritizing closed-loop autonomous world operation and simpler code.

At the end of a meaningful session, update this section with only the current
state or a concise new decision. Do not keep a long historical log.
