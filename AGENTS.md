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

Remove dead code, unnecessary tests, speculative configuration, and abstractions
that do not provide present value. Preserve unrelated user work. Refactor
broadly when it makes the whole system cleaner.

Continue iterating toward the project goal autonomously. When no meaningful
work remains, sleep for 15 minutes, reread `STEERING.md`, and continue; repeat
until new steering is given.

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

Keep movement slow, deliberate, and easy to stop.

## Simulation and validation

Gazebo with PX4 SITL is the primary development environment and the authority
for software flight behavior. Develop, iterate, and review the system there so
most progress requires no physical flight. A meaningful scenario must be
confirmed from output or telemetry, including connection, readiness, arming,
offboard setpoints, expected motion, safety intervention, landing, and disarm.

Do not add unit tests. Keep verification in Gazebo and through end-to-end
behavior. Remove existing unit tests when their behavior is covered by the
simulation; do not preserve code or architecture to support them.

From `companion/`, the normal simulation loop is:

```sh
PYTHONPYCACHEPREFIX=/tmp/companion-pycache .venv/bin/python -m compileall -q control onboard sim vision voice tests
python -m sim.command_loopback
python -m sim.run_world
python -m sim.run_world --image .venv/lib/python3.9/site-packages/ultralytics/assets/bus.jpg
```

`sim.run_world` manages PX4/Gazebo and cleanup. The synthetic world exercises
the command path, motion, target loss, obstacle handling, malformed input,
dropout, recovery, shutdown, landing, and disarm. The image scenario exercises
decoded RTP/video, perception, Mac commands, onboard safety, and PX4. Neither
scenario proves physical sensors, radio behavior, or hardware-specific
transport.

Fix failures at their source. Do not weaken simulation assertions or add
retries merely to make a run pass.

## Current state

Simulation is the main development environment, not a preliminary demo. Keep
building thorough, autonomous PX4/Gazebo scenarios that exercise the complete
control path, varied perception and sensor conditions, faults, recovery,
safety, and long-running behavior. Use those scenarios to develop, refactor,
and review the companion continuously without physical intervention.

Hardware bring-up later validates how well the simulated interfaces translate
to the real camera, network, sensors, and vehicle. It should add evidence and
calibration, not replace the simulation-first development loop.
