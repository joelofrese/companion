# Companion Drone

Keep this guide accurate, simple, and easy to edit.

## Goal

Build an autonomous indoor companion drone that notices, decides, moves
deliberately, stays safe, and learns over time.

Develop and test it in simulation whenever possible so progress does not
depend on hardware.

## Priorities

1. Readable code and design.
2. Simple code and design.
3. Minimal code and design.

This keeps the system easy to understand, debug, develop, and maintain.

Remove dead code and speculative abstractions. Refactor broadly when it makes
the whole system simpler. Keep reviewing, simplifying, and developing the
companion without waiting for confirmation. When no meaningful work remains,
sleep for 15 minutes, reread this guide, and continue.

## Git

Work directly on `main` in this single-contributor repository. Push verified
checkpoints there. Use a temporary branch only for risky isolated work, then
merge and delete it.

## Control

- One Gemini ER 2 Streaming session starts with one situation prompt and keeps
  deciding from the newest image, dialogue, local position, heading, other
  telemetry, memory, and action results.
- A new user request is the active task until it is completed, changed, or
  unsafe. Explore generally only when there is no more specific request.
- Gemini chooses direct `move`, `turn`, `hover`, or `speak` tools. Text or JSON
  action descriptions never move the vehicle.
- `move` is a short, slow body-frame pulse. It may include vertical velocity or
  a small yaw rate for a smooth arc. `turn` accepts a relative angle and settles
  from heading.
- Physical move and turn calls are blocking in the ER 2 robotics session. The
  runtime returns measured motion, heading, fresh telemetry, and a newer camera
  frame before another movement is chosen. Move results also report measured
  local-position change. An explicit stop may interrupt an action.
- Choose each action from the newest image and measured state. Use short,
  deliberate pulses, inspect the fresh result, and stop when no useful change
  is clear. The forward-only TOF reading calls for a safe lateral, backward,
  or turning response when the path is blocked. Do not narrate routine
  movement. After repeated in-place turns without a translation, translate or
  wait for new dialogue before turning again.
- Camera frames stream once per second. Heartbeats continue sending frames while
  a physical action runs, but the blocking tool keeps ER 2 from choosing another
  action until its result returns. Dialogue waits for that result unless it is an
  explicit stop; a bounded timeout restarts a stalled session.
- Stale, malformed, missing, or unsafe input becomes zero motion. The CM5
  rejects unsafe commands, stops positive forward motion at a close obstacle,
  and is the final vehicle-side authority. PX4 stabilizes the vehicle and
  controls the motors.

The brain sends only slow body-frame translation and yaw-rate commands. Vertical
translation is a velocity pulse, never an altitude command. It never sends
motor, attitude, or absolute-position commands.

## Hardware

The CM5 runs the camera, Gemini session, brain, final safety checks, and PX4
forwarding. It sends only approved body-frame velocity and yaw-rate setpoints,
using fresh vehicle heading for translation. A Mac is optional for simulation,
development, and remote operation.

The target is the DroneBlocks DEXI 3: PX4, optical flow, a TOF distance sensor,
a Raspberry Pi camera, and a Raspberry Pi CM5. It has no lidar. Keep
simulation-only sensors separate from this hardware description.

## Simulation

PX4 SITL with Gazebo is the primary development environment and authority for
software flight behavior. Exercise the full control path, perception, varied
worlds, faults, recovery, safety, long runs, landing, and disarm. Verify actual
output or telemetry for connection, readiness, arming, setpoints, motion,
safety intervention, local position, landing, and disarm.

Do not add unit tests. Prefer small end-to-end checks and real simulator
behavior so the code stays flexible.

Keep two modes:

- Deterministic missions prove transport, perception fixtures, flight, faults,
  recovery, safety, landing, and disarm.
- Exploratory worlds give Gemini an open situation and let it choose what to
  do. Verify bounded motion, safety, landing, and disarm rather than exact
  decisions.

The deterministic brain is simulation-only; Gemini is the only production
visual and decision model. The fixture exists only for repeatable control-path
checks.

From `companion/`:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice
.venv/bin/python -m sim.command_loopback
.venv/bin/python -m sim.run_world
.venv/bin/python -m sim.run_world --explore --depth --gemini --trace --world objects --duration 60
.venv/bin/python -m sim.run_world --explore --depth --gemini --trace --world walls --duration 60
.venv/bin/python -m sim.run_world --explore --faults --depth --gemini --trace --world objects --duration 32
.venv/bin/python -m sim.run_world --image <image-path> --expect-person
```

`sim.run_world` manages PX4, Gazebo, cleanup, and exploratory dialogue. Use
`--request`, `--intent`, `--memory`, and typed dialogue to vary a run. Use
`--trace` to see camera/telemetry state, ER 2 thought summaries when provided,
responses, tool calls, latencies, and command reasons. Raw private reasoning is
not exposed. Use `--headless` for unattended runs and `--snapshot PATH` to
save a rendered frame.

`--gemini` uses the streaming ER 2 brain. `--camera` uses Gazebo video;
`--depth` adds simulated forward range for CM5 safety and allows movement.
Camera-only runs stop because they have no range reading. `--faults` injects
sensor, camera, command-link, brain, and Gemini reconnect faults.

The companion `objects` world contains simple furniture, colored shapes, a
mannequin, and a central obstacle. Other exploratory runs use stock Gazebo
worlds. `--moving-person` moves the mannequin through Gazebo's pose service.
This is a visual fixture, not a DEXI 3 hardware claim. The depth model is only
an approximation of DEXI 3's forward TOF sensor.

The image scenario verifies RTP transport through the deterministic fixture;
Gazebo camera explorations verify live visual behavior through Gemini. No local
visual decision model remains beside Gemini.

## Running on hardware

```sh
.venv/bin/python -m control.companion <cm5-ip>
.venv/bin/python -m control.companion <cm5-ip> --dialogue

# On the CM5, with onboard.ros2_bridge already running:
.venv/bin/python -m control.companion --local --dialogue
```

Set `GEMINI_API_KEY`. The brain uses one persistent ER 2 Streaming session
with native context-window compression and session resumption for ordinary
disconnects. A silent model turn restarts from the situation and memory. Its
editable memory file contains only prior experience across runs. The newest
640-pixel JPEG, telemetry, dialogue, and action results remain the live context;
recent measured actions are retained when a fresh session is needed.

## Current state

- Deterministic PX4/Gazebo missions and local UDP loopback verify the command
  path, telemetry, faults, recovery, safety, landing, and disarm.
- Exploratory stock and companion-owned worlds exercise open-ended ER 2
  decisions, dialogue, memory, movement, and simulated TOF safety.
- ER 2 chooses movement, turn, hover, or speech tools.
  Physical move and turn calls block until measured completion and heading; move
  results expose numeric measured translation or angle and local-position or
  heading feedback for calibration.
- Native ER 2 thinking is currently set low for timely closed-loop control;
  latency and decisions remain variable.
- CM5 limits every physical command and PX4 stabilizes the vehicle. Hardware
  behavior remains unverified. DEXI 3 has no lidar.

At the end of a meaningful session, update this section only with the current
state or one concise decision. Do not keep a history here.
