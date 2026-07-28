# Companion Drone Project

> **Living document** — update this file at the end of each Codex session with progress, decisions, and any new context. Suggest changes when something seems off or incomplete.

## Code Quality

Keep every change as simple, minimal, readable, and maintainable as possible. These qualities take priority over feature speed and breadth. Working behavior is only the first completion gate; the final implementation must also be the simplest coherent design.

- Optimize for the simplest final codebase, not the smallest immediate diff.
- Write the smallest clear solution that fully solves the current problem.
- Prefer straightforward code over cleverness, abstraction, configurability, or premature extensibility.
- Add dependencies, files, layers, and configuration only when they provide immediate, concrete value.
- Reuse existing code when it remains clear; remove duplication only when the resulting abstraction is simpler.
- Delete obsolete code and comments instead of preserving them for hypothetical future use.
- Keep functions focused, names precise, control flow obvious, and public interfaces small.
- Avoid speculative features, compatibility layers, fallback paths, and defensive complexity unless the project currently requires them.
- Treat code review as a simplification pass: remove anything unnecessary before considering a change complete.
- Safety-critical behavior is not optional complexity. Keep it explicit, small, and easy to verify.

## Autonomous Development

Work autonomously without waiting for user input. Make product, architecture, prioritization, implementation, and in-scope maintenance decisions according to the project vision and constraints. When several reasonable choices exist, choose the simplest, safest, most reversible option, document consequential decisions, create a Git checkpoint, and continue.

Refactor freely when the existing structure would make a change less clear. Preserve required behavior, safety properties, and architectural boundaries—not incidental implementation details.

- Take initiative beyond the current checklist. Identify and implement aligned capabilities that would make the companion meaningfully more capable, natural, or useful without waiting for preapproval.
- Prefer changing, consolidating, or deleting existing code over layering new code around it.
- Rewrite modules, change internal interfaces, and update all callers together when that produces a simpler whole.
- Do not retain obsolete paths, temporary adapters, or compatibility shims after the change that required them is complete.
- Avoid unrelated aesthetic churn. Every broad refactor must have a concrete simplification or correctness goal.
- Establish a working baseline before substantial restructuring and verify the same relevant behavior afterward.
- Implement the smallest complete vertical slice, then perform a separate simplification pass across the final diff.
- Consider added dependencies, files, configuration, processes, public interfaces, and abstractions to be maintenance costs that require present value.
- Preserve existing safety constraints when uncertain and continue in simulation rather than weakening safeguards.
- If one task is blocked, record why and continue with another valuable part of the project. Stop only when no meaningful work can proceed; platform-required permissions, unavailable credentials, or unavailable hardware may still block specific actions.

Use Git as a recovery mechanism, not as a reason to avoid worthwhile refactoring.

- Preserve unrelated and uncommitted user work; never include it in a checkpoint.
- Use a feature branch rather than working directly on the protected branch.
- Create cohesive checkpoints before risky restructuring and whenever the repository returns to a verified, meaningful state.
- Push stable checkpoints to the feature branch when remote access is available, but never force-push or rewrite shared history.
- Keep commits understandable and reversible; avoid both sprawling mixed-purpose commits and noisy microcommits.
- Run the relevant tests or simulator before calling a milestone complete. If an approach fails, return to the last known-good checkpoint rather than accumulating workarounds.

## Testing and Simulation

Develop and run the relevant tests and simulated scenarios autonomously as part of implementation.

- Use focused automated tests for deterministic logic and PX4 SITL/Gazebo for integrated flight behavior.
- Treat test and simulation code as production code: keep it small, readable, deterministic, and free of unnecessary harnesses, mocks, fixtures, and configuration.
- Test observable behavior and safety properties rather than internal implementation details so refactoring remains easy.
- Extend the simulation setup only when a current behavior needs it; reuse the existing launch and verification paths when they remain clear.
- Validate flight behavior in simulation before hardware testing. Confirm expected events and outcomes from actual output or telemetry rather than process survival.
- Fix failures at their source. Do not weaken assertions, skip checks, or add retries merely to make a test pass.
- Consider a change complete only after the relevant checks pass and the implementation and its tests have both received a simplification pass.

## Vision
Build a companion drone that operates like a living thing — the AI sets high-level intent (like a brain deciding to look at something), while lower layers handle the automatic execution (like reflexes and muscle memory). The goal is natural, creature-like behavior rather than robotic point-to-point movement.

## Hardware
- **Drone:** DroneBlox DEXI 3
  - Flight controller: PX4 Autopilot (FMUv6X)
  - Onboard compute: Raspberry Pi CM5
  - Sensors: Optical flow, TOF distance sensor, Pi camera
  - Positioning: Optical flow for indoor/GPS-denied flight
- **Brains:** Mac (all heavy compute runs here, not on drone)
- **Link:** WiFi — drone streams data to Mac, Mac sends commands back

## Architecture: 3-Layer Control

Inspired by biological motor control. Each layer only talks to the one directly below it — nothing skips levels.

| Layer | Biological analogy | What runs it | Speed | Responsibility |
|---|---|---|---|---|
| **Cognitive** | Cerebral cortex | LLM + vision AI on Mac | ~1–5 Hz | Sets intent/state |
| **Reactive** | Cerebellum / reflexes | Python on Mac or CM5 | ~20–50 Hz | Executes state as velocity commands, handles safety |
| **Stabilization** | Spinal cord / muscles | PX4 (automatic) | ~400 Hz | Attitude, motor mixing |

**Key principle:** The AI never directly controls motors or sends raw velocity commands. It only transitions between states. Each state's logic handles the actual `vx/vy/vz` output.

## Data Flow

```
Mac (brains)                        Drone (body)
┌──────────────────────┐            ┌─────────────────┐
│  Vision model (YOLO) │ ← video ───│  Pi CM5 camera  │
│  LLM / interaction   │ ← audio ───│  Microphone     │
│  State machine       │ ← telemetry│  PX4 sensors    │
│  Kalman filter       │            │                 │
│                      │─ commands ▶│  PX4 via MAVSDK │
└──────────────────────┘   WiFi     └─────────────────┘
```

- Video: compressed H.264, ~480p, streamed over local WiFi
- WiFi round-trip latency: ~10–30ms (negligible)
- **Safety rule:** Reactive obstacle-stop layer stays functional even if WiFi drops

## State Machine

The AI transitions between states. Each state runs its own control loop producing velocity setpoints.

```
IDLE → FOLLOWING → AVOIDING → HOVERING → RESPONDING → IDLE
```

- **IDLE:** Hovering in place, minimal movement, listening
- **FOLLOWING:** Tracking a person; reactive layer computes vx/vy to close distance slowly
- **AVOIDING:** Obstacle detected via TOF; overrides other states, backs off
- **HOVERING:** Hold position, waiting for next intent
- **RESPONDING:** Reacting to voice/gesture input (turning to face speaker, etc.)

States are set by the AI. Transitions happen slowly (1–5 Hz) — that's fine.

## Motion Control

- **Command type:** Velocity setpoints (`vx, vy, vz`) in PX4 offboard mode — never absolute position commands
- **Why velocity:** Smooth, cancellable mid-move, safe zero-velocity fallback (just hover)
- **Person tracking:** Kalman filter predicts position ~300ms ahead to compensate for inference latency
- **Speed:** Intentionally slow. Creature-like deliberate movement is the goal, not speed.

## Tech Stack

| Task | Tool |
|---|---|
| Flight control | PX4 + MAVSDK-Python (offboard velocity mode) |
| Vision / detection | OpenCV + YOLOv8n (nano — fast enough on Mac, low latency) |
| State machine | Python `transitions` library |
| Position prediction | `filterpy` Kalman filter |
| Voice interaction | Whisper (on Mac) → parsed intent → state command |
| Simulation | Gazebo + PX4 SITL (for development before hardware arrives) |
| Onboard OS | DEXI-OS (Raspberry Pi CM5) |

## Development Notes

- **Launch sim:** Run `sim/launch_sim.sh` (or `make px4_sitl gz_x500` from `~/Code/Croppie/PX4-Autopilot` with venv activated). QGC is not required. Run scripts from `companion/` with its own `.venv`.
- **Simulate first:** The full cognitive + reactive stack can be tested in Gazebo with a virtual camera feed before the drone arrives. Validate state transitions, velocity behavior, and AI loop timing in simulation.
- **No GPU needed on drone:** CM5 is sensor aggregator + command executor only. All inference runs on Mac.
- **Offboard mode:** PX4 requires a continuous stream of setpoints in offboard mode — if commands stop, it will hover or land. Design the reactive layer to always be sending something.
- **DEXI-OS repos:** DroneBlocks maintains DEXI-OS and a ROS2 repository — check these for CM5 setup and GPIO access.
- **Verifying sim runs:** Launch PX4 SITL unbuffered so progress is actually visible in a log (`stdbuf -oL -eL make px4_sitl gz_x500`, output otherwise buffers indefinitely when redirected to a file/pipe) — wait for `pxh>` in the log as the boot-complete marker. Run `sim/hover.py` with `python -u` for the same reason. Confirm a real pass by tailing the script's stdout for the full sequence: `Connected.` → `Ready.` → `Arming...` → `Taking off...` → `Landed.` — don't infer success just from processes staying alive.
  - **Known flaky boot:** PX4's `px4-rc.mavlink` script starts 4 MAVLink links (GCS on 18570, Onboard API on 14580→14540, payload, gimbal). Occasionally only the GCS link comes up and the Onboard link — the one MAVSDK actually connects to — silently never starts, with no error printed. Symptom: `hover.py` hangs forever at "Waiting for drone connection..." with 0% CPU. Diagnose with `lsof -p <px4_pid> -i UDP` and check for a socket on 14580; if it's missing, kill everything (`pkill -9 -f "bin/px4 "`, `pkill -9 -f "gz sim"`, `pkill -9 -f mavsdk_server`) and relaunch — it came up clean on retry.
  - Also watch for orphaned `mavsdk_server` subprocesses: killing `hover.py`'s parent Python process doesn't kill its `mavsdk_server` child, which keeps holding UDP 14540 and causes `bind error: Address already in use` on the next run.

## Project Status
- [x] PX4 SITL + Gazebo simulation environment set up
- [x] MAVSDK-Python connection plus arm/takeoff/hover/land actions working in simulation
- [ ] PX4 offboard mode with continuous MAVSDK velocity setpoints working in simulation
- [ ] State machine skeleton implemented
- [ ] YOLOv8n person detection pipeline working
- [ ] Kalman filter for person tracking implemented
- [ ] WiFi video stream from CM5 to Mac working
- [ ] Voice command pipeline (Whisper → intent → state)
- [ ] Full integration test in simulation
- [ ] Hardware arrives (DroneBlox DEXI 3)
- [ ] First real flight test

## Open Questions
- **Reactive layer location:** Does the reactive layer (20–50 Hz control loop) run on the Mac or on the CM5? Mac gives more compute; CM5 survives WiFi drops. Current lean: Mac, with a simple heartbeat-based failsafe on CM5.
- **Obstacle avoidance scope:** TOF sensor gives a single forward distance. Is that enough, or do we need multi-directional sensing for the avoiding state?
- **State transition authority:** Can the reactive layer ever trigger a state transition (e.g. force AVOIDING), or does only the cognitive layer set state? Needs a clear rule to avoid race conditions.
- **Voice trigger:** Always-on listening vs. push-to-talk? Affects Whisper latency and battery tradeoffs.

## Resolved Decisions
- **DEXI-OS:** Use it. The CM5→PX4 serial/UART wiring, MAVLink routing, camera drivers, and optical flow sensor config are all pre-done. No reason to start from a blank Pi OS — it doesn't constrain the Mac-side architecture at all.
- **Power and tethering:** Do not build a custom power tether or modify the drone's power electronics. Accept the stock battery's short flight time, do most development in simulation, and use brief untethered flights for hardware validation.

## Constraints / Non-Goals
- **No GPS** — positioning is optical flow only; designed for indoor/GPS-denied environments
- **No direct motor control** — the AI never sends raw motor commands; PX4 handles all stabilization
- **No position setpoints** — only velocity setpoints (`vx, vy, vz`); absolute position commands are explicitly ruled out
- **No GPU on drone** — all ML inference (YOLO, Whisper) stays on the Mac; CM5 is strictly a sensor relay and command executor
- **Not fast** — speed is explicitly deprioritized; creature-like deliberate movement is the goal
- **No layer skipping** — cognitive layer never bypasses reactive layer to talk directly to PX4

## Session Log
- **2026-05-24** — Project started. Transferred the architecture and vision from the initial design conversation into the project guide. No code written yet. Hardware not yet arrived.
- **2026-05-25** — Full sim environment working: PX4 SITL + Gazebo Harmonic running on macOS, no QGC required. `sim/hover.py` successfully arms, takes off to 2m, hovers 30s, and lands via MAVSDK. `libGstCameraSystem.dylib` now loads cleanly: root cause was GTK3/GTK4 conflict from GStreamer's plugin scanner loading `libgstgtk.dylib` + `libgstgtk4.dylib` — fixed by removing those GTK plugin dylibs from GStreamer's plugin dir and rebuilding PX4 to recompile the camera system plugin. `DYLD_LIBRARY_PATH=/opt/homebrew/lib` set in launch script to silence GLib introspection warnings from gst-plugin-scanner. Health check in hover.py waits for magnetometer calibration in addition to GPS — otherwise arm is rejected with "no heading reference". SDF warnings about `gz_frame_id` are harmless. Protobuf 6.x `Resize`→`resize` fix applied in GZMixingInterfaceESC.cpp and GZMixingInterfaceWheel.cpp.
- **2026-07-13** — Picked project back up after a couple months away. Re-verified `sim/hover.py` still works end-to-end (see Development Notes above for how). Hit and resolved the flaky Onboard-MAVLink-link boot issue and an orphaned `mavsdk_server` port conflict along the way — both now documented above so future sessions don't have to re-diagnose them. Bumped `HOVER_DURATION` to 15s for easier observation.
- **2026-07-26** — Audited the repository after migrating its project guide to Codex. The Python environment, MAVSDK import, script syntax, PX4 checkout, and existing SITL build are present. Clarified that `sim/hover.py` proves MAVSDK connectivity and high-level action flight, but does not yet exercise PX4 offboard mode or velocity setpoints.
- **2026-07-26** — Established simplicity and minimalism as permanent code-quality requirements for every future change, while keeping safety-critical behavior explicit and verifiable.
- **2026-07-26** — Ruled out a physical power tether and custom power electronics. Development will remain simulation-first, with short stock-battery flights used only when hardware validation is necessary.
- **2026-07-26** — Re-ran the complete PX4 SITL/Gazebo hover test successfully: MAVSDK connected, the simulated x500 armed, took off to 2 m, hovered for 15 seconds, landed, and disarmed. The documented SDF and missing-GCS warnings remained harmless, and the simulator shut down cleanly afterward.
- **2026-07-28** — Completed the project-guide migration from `CLAUDE.md` to `AGENTS.md` and established autonomous development: make aligned product and technical decisions without interruption, refactor toward the simplest final system, originate useful new capabilities, use verified Git checkpoints, favor safe and reversible choices, and continue with other valuable work when an individual task is blocked.
- **2026-07-28** — Made simplicity, readability, and maintainability higher priorities than feature throughput, and made autonomous, maintainable automated testing plus SITL/Gazebo verification part of the definition of done.
