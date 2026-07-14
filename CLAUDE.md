# Companion Drone Project

> **Living document** — update this file at the end of each session with progress, decisions, and any new context. Suggest changes when something seems off or incomplete.

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
- **Verifying sim runs (for Claude or anyone else):** Launch PX4 SITL unbuffered so progress is actually visible in a log (`stdbuf -oL -eL make px4_sitl gz_x500`, output otherwise buffers indefinitely when redirected to a file/pipe) — wait for `pxh>` in the log as the boot-complete marker. Run `sim/hover.py` with `python -u` for the same reason. Confirm a real pass by tailing the script's stdout for the full sequence: `Connected.` → `Ready.` → `Arming...` → `Taking off...` → `Landed.` — don't infer success just from processes staying alive.
  - **Known flaky boot:** PX4's `px4-rc.mavlink` script starts 4 MAVLink links (GCS on 18570, Onboard API on 14580→14540, payload, gimbal). Occasionally only the GCS link comes up and the Onboard link — the one MAVSDK actually connects to — silently never starts, with no error printed. Symptom: `hover.py` hangs forever at "Waiting for drone connection..." with 0% CPU. Diagnose with `lsof -p <px4_pid> -i UDP` and check for a socket on 14580; if it's missing, kill everything (`pkill -9 -f "bin/px4 "`, `pkill -9 -f "gz sim"`, `pkill -9 -f mavsdk_server`) and relaunch — it came up clean on retry.
  - Also watch for orphaned `mavsdk_server` subprocesses: killing `hover.py`'s parent Python process doesn't kill its `mavsdk_server` child, which keeps holding UDP 14540 and causes `bind error: Address already in use` on the next run.

## Project Status
- [x] PX4 SITL + Gazebo simulation environment set up
- [x] MAVSDK-Python velocity control working in simulation
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

## Constraints / Non-Goals
- **No GPS** — positioning is optical flow only; designed for indoor/GPS-denied environments
- **No direct motor control** — the AI never sends raw motor commands; PX4 handles all stabilization
- **No position setpoints** — only velocity setpoints (`vx, vy, vz`); absolute position commands are explicitly ruled out
- **No GPU on drone** — all ML inference (YOLO, Whisper) stays on the Mac; CM5 is strictly a sensor relay and command executor
- **Not fast** — speed is explicitly deprioritized; creature-like deliberate movement is the goal
- **No layer skipping** — cognitive layer never bypasses reactive layer to talk directly to PX4

## Session Log
- **2026-05-24** — Project started. Transferred architecture and vision from Claude chat into CLAUDE.md. No code written yet. Hardware not yet arrived.
- **2026-05-25** — Full sim environment working: PX4 SITL + Gazebo Harmonic running on macOS, no QGC required. `sim/hover.py` successfully arms, takes off to 2m, hovers 30s, and lands via MAVSDK. `libGstCameraSystem.dylib` now loads cleanly: root cause was GTK3/GTK4 conflict from GStreamer's plugin scanner loading `libgstgtk.dylib` + `libgstgtk4.dylib` — fixed by removing those GTK plugin dylibs from GStreamer's plugin dir and rebuilding PX4 to recompile the camera system plugin. `DYLD_LIBRARY_PATH=/opt/homebrew/lib` set in launch script to silence GLib introspection warnings from gst-plugin-scanner. Health check in hover.py waits for magnetometer calibration in addition to GPS — otherwise arm is rejected with "no heading reference". SDF warnings about `gz_frame_id` are harmless. Protobuf 6.x `Resize`→`resize` fix applied in GZMixingInterfaceESC.cpp and GZMixingInterfaceWheel.cpp.
- **2026-07-13** — Picked project back up after a couple months away. Re-verified `sim/hover.py` still works end-to-end (see Development Notes above for how). Hit and resolved the flaky Onboard-MAVLink-link boot issue and an orphaned `mavsdk_server` port conflict along the way — both now documented above so future sessions don't have to re-diagnose them. Bumped `HOVER_DURATION` to 15s for easier observation.
