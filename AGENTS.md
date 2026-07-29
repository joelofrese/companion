# Companion Drone

This is a short, living guide. Keep it accurate, simple, and editable. Record
only decisions, constraints, and evidence that future work needs.

## Goal

Build an autonomous indoor companion drone that behaves naturally: it should
notice, decide, move deliberately, stay safe, and remain useful without a
human translating every observation into the next action.

The system should be freely developed in simulation first and later refined
with hardware evidence. Simulation may stand in for missing hardware, but must
not be described as proof of hardware behavior.

## Priorities

1. Safety and observable behavior.
2. The simplest coherent design.
3. Readability and easy refactoring.
4. Useful capability over feature count.

Remove dead code, redundant tests, speculative configuration, and abstractions
that do not provide present value. Preserve safety properties and unrelated
user work. Refactor broadly when it makes the whole system clearer.

Work autonomously: choose the next valuable, reversible step, verify it, and
checkpoint stable progress on the feature branch. If hardware or credentials
block one path, continue with useful simulation. Read `STEERING.md` once when
starting a meaningful milestone, if it exists; do not poll it during work.

## Control boundary

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
override normal intent. Safety behavior must be explicit and bounded.

## System boundary

Heavy perception, cognition, and interaction run on the Mac. The vehicle-side
computer relays sensors and video, enforces the final command boundary, and
forwards approved velocity setpoints to PX4. Hardware-specific integration is
kept at that boundary so the Mac-side control behavior remains testable.

Do not add hardware features, sensing directions, speed, or autonomy claims
without a concrete need and evidence. Prefer deliberate, cancellable motion.

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

## Current boundary

The software control and simulation paths are implemented and continuously
verified. Hardware bring-up remains the next source of evidence for camera and
network behavior, onboard sensor data, flight-controller forwarding, and real
flight. Keep those unknowns explicit rather than hiding them behind simulation.
