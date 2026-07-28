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

- Reread the root `AGENTS.md` before selecting each new milestone, after every meaningful Git checkpoint, and immediately after modifying it. Incorporate the latest guidance without pausing for user confirmation.
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
- Keep only tests that protect distinct behavior, safety properties, or active integration contracts. A larger test count is not progress; remove redundant coverage across layers.
- Test observable behavior and safety properties rather than internal implementation details so refactoring remains easy.
- Prefer direction, bounds, invariants, and tolerances over exact numeric assertions for tunable motion behavior. Assert exact values only when they are deliberate requirements or safety limits.
- Refactor or delete tests with the code they describe. Never preserve an obsolete interface or architecture merely to keep an old test passing.
- Extend the simulation setup only when a current behavior needs it; reuse the existing launch and verification paths when they remain clear.
- Validate flight behavior in simulation before hardware testing. Confirm expected events and outcomes from actual output or telemetry rather than process survival.
- Treat unit tests as fast supporting evidence, not a substitute for real adapter checks or SITL/Gazebo behavior. Integrated simulation is the final authority for flight behavior.
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
| State machine | Dependency-free Python state machine |
| Position prediction | Dependency-free constant-velocity Kalman tracker |
| Voice interaction | Whisper (on Mac) → parsed intent → state command |
| Simulation | Gazebo + PX4 SITL (for development before hardware arrives) |
| Onboard OS | DEXI-OS (Raspberry Pi CM5) |

## Development Notes

- **Launch sim:** Run `sim/launch_sim.sh` (or `make px4_sitl gz_x500` from `~/Code/Croppie/PX4-Autopilot` with venv activated). QGC is not required. Run scripts from `companion/` with its own `.venv`.
- **Simulate first:** The full cognitive + reactive stack can be tested in Gazebo with a virtual camera feed before the drone arrives. Validate state transitions, velocity behavior, and AI loop timing in simulation.
- **No GPU needed on drone:** CM5 is sensor aggregator + command executor only. All inference runs on Mac.
- **CM5 video sender:** Once DEXI-OS is available, run `python -m onboard.video_sender <mac-ip>` on the CM5. The wrapper owns the libcamera→RTP child process and accepts optional `--port`, `--width`, `--height`, and `--framerate` overrides; this host cannot execute it because `libcamerasrc` is unavailable.
- **CM5 command receiver:** The future DEXI-OS PX4 forwarder should poll `onboard.command_receiver.UdpSafetyReceiver` at 20–50 Hz and forward only its returned `VelocityCommand`. `python -m sim.command_loopback` verifies the packet and local safety path over UDP; it does not claim a real Wi-Fi or PX4-forwarding test.
- **CM5 command service:** `onboard.command_service.SafetyCommandService` owns the receiver lifecycle, fixed-rate safety polling, approved-command forwarding, and one explicit zero command on shutdown. Call `start()` synchronously before the Mac sender can transmit; `onboard.ros2_bridge` supplies the DEXI ROS 2 forwarder and distance callback at the hardware boundary.
- **CM5 ROS 2 bridge:** On DEXI-OS with `rclpy` and `px4_msgs` installed, run `python -m onboard.ros2_bridge`. It binds UDP before starting the service thread, subscribes to `/fmu/out/distance_sensor`, initializes the distance reading as unsafe until a message arrives, expires it after 150 ms without a fresh message, and publishes velocity-only PX4 setpoints with the injected ROS clock. The bridge owns `/fmu/in/offboard_control_mode` and `/fmu/in/trajectory_setpoint`; do not run the stock `px4_offboard_manager` concurrently because it publishes position-mode setpoints. The bridge is not executable on this Mac because ROS 2 is not installed here.
- **CM5 command bounds:** `onboard.safety.OnboardSafetyEnvelope` independently accepts at most `0.5 m/s` per horizontal NED component and `0.3 m/s` vertical; a fresh command outside those limits or with non-finite fields becomes zero. These limits match the current Mac follower’s deliberate speed envelope.
- **DEXI ROS 2 transport:** The public DroneBlocks `dexi` repository's current launch path starts `micro_ros_agent serial --dev /dev/ttyAMA2 -b 3000000` and a `px4_offboard_manager` node. `onboard.ros2_forwarder.Ros2VelocityForwarder` is the companion's hardware-neutral publisher seam for that path; it emits velocity-only `OffboardControlMode` and `TrajectorySetpoint` messages without importing ROS on the Mac.
- **UDP-to-SITL verification:** Run `python -m sim.offboard_udp` against PX4 SITL to exercise Mac packet generation → CM5 safety receiver → MAVSDK NED forwarding. The verifier includes a 150 ms heartbeat dropout, a deliberate out-of-bounds packet, and a forward obstacle; the MAVSDK sink is a simulation stand-in, to be replaced with the DEXI-OS MAVLink/PX4 process when hardware arrives.
- **Offboard mode:** PX4 requires a continuous stream of setpoints in offboard mode — if commands stop, it will hover or land. Design the reactive layer to always be sending something.
- **DEXI-OS repos:** DroneBlocks maintains DEXI-OS and a ROS2 repository — check these for CM5 setup and GPIO access.
- **Verifying sim runs:** Launch PX4 SITL unbuffered so progress is actually visible in a log (`stdbuf -oL -eL make px4_sitl gz_x500`, output otherwise buffers indefinitely when redirected to a file/pipe) — wait for `pxh>` in the log as the boot-complete marker. Run `sim/hover.py` with `python -u` for the same reason. Confirm a real pass by tailing the script's stdout for the full sequence: `Connected.` → `Ready.` → `Arming...` → `Taking off...` → `Landed.` — don't infer success just from processes staying alive.
  - **Known flaky boot:** PX4's `px4-rc.mavlink` script starts 4 MAVLink links (GCS on 18570, Onboard API on 14580→14540, payload, gimbal). Occasionally only the GCS link comes up and the Onboard link — the one MAVSDK actually connects to — silently never starts, with no error printed. Symptom: `hover.py` hangs forever at "Waiting for drone connection..." with 0% CPU. Diagnose with `lsof -p <px4_pid> -i UDP` and check for a socket on 14580; if it's missing, kill everything (`pkill -9 -f "bin/px4 "`, `pkill -9 -f "gz sim"`, `pkill -9 -f mavsdk_server`) and relaunch — it came up clean on retry.
  - Also watch for orphaned `mavsdk_server` subprocesses: killing `hover.py`'s parent Python process doesn't kill its `mavsdk_server` child, which keeps holding UDP 14540 and causes `bind error: Address already in use` on the next run.

## Project Status
- [x] PX4 SITL + Gazebo simulation environment set up
- [x] MAVSDK-Python connection plus arm/takeoff/hover/land actions working in simulation
- [x] PX4 offboard mode with continuous MAVSDK velocity setpoints working in simulation
- [x] State machine skeleton implemented
- [x] YOLOv8n person detection pipeline working
- [x] Kalman filter for person tracking implemented
- [ ] WiFi video stream from CM5 to Mac working
- [x] Voice command pipeline (Whisper → intent → state)
- [x] Full integration test in simulation (deterministic voice intent + decoded RTP video + reactive obstacle safety)
- [ ] Hardware arrives (DroneBlox DEXI 3)
- [ ] First real flight test

## Open Questions
- **CM5 PX4 forwarding:** The full reactive layer stays on the Mac; the CM5 now owns the final heartbeat/TOF safety envelope and a versioned command packet codec. `onboard.ros2_bridge` is the concrete ROS 2 process seam for DEXI-OS, but hardware bring-up still needs validation of its serial transport, distance-sensor topic, and PX4 response.
- **Obstacle avoidance scope:** TOF sensor gives a single forward distance. Is that enough, or do we need multi-directional sensing for the avoiding state? The public DEXI ROS 2 repository does not expose a TOF driver or distance topic, so the CM5 sensor wiring and topic remain hardware-gated.

## Resolved Decisions
- **DEXI-OS:** Use it. The CM5→PX4 serial/UART wiring, MAVLink routing, camera drivers, and optical flow sensor config are all pre-done. No reason to start from a blank Pi OS — it doesn't constrain the Mac-side architecture at all.
- **Power and tethering:** Do not build a custom power tether or modify the drone's power electronics. Accept the stock battery's short flight time, do most development in simulation, and use brief untethered flights for hardware validation.
- **Voice trigger:** Use push-to-talk for the first implementation. It avoids accidental commands and keeps idle CPU/battery use predictable; always-on listening can be reconsidered after the core flight loop is safe.
- **Visual follow geometry:** Start with a forward-facing 640 px camera model: target height controls north-frame distance correction and horizontal image error controls east-frame correction. Calibrate camera orientation and desired target size before hardware flight.
- **Reactive layer location:** Keep vision, tracking, state, and normal velocity generation on the Mac; run only the small CM5 safety envelope locally so Wi-Fi loss cannot preserve a stale command or bypass the forward obstacle stop.
- **Command wire contract:** Use version-1 compact JSON packets containing a non-negative sequence and NED/yaw velocity; timestamp on CM5 receipt rather than synchronizing Mac and CM5 clocks. Duplicate or reordered sequence numbers do not refresh the heartbeat.
- **State transition authority:** Cognitive intent owns the persistent state; reactive obstacle safety may expose a transient `AVOIDING` state and backoff command, then restores the saved intent when the path is clear. Safety can override output, but it does not permanently rewrite cognitive intent.
- **CM5 service lifecycle:** Bind the receiver before the Mac sends its first packet, run safety polling at the reactive setpoint rate, and send a zero velocity during orderly shutdown. The service remains transport-independent; `onboard.ros2_bridge` is the DEXI-OS PX4 forwarder composition.
- **ROS 2 flight setpoints:** Keep the DEXI integration velocity-only even though the stock `px4_offboard_manager` also contains position-navigation helpers. The companion bridge replaces that manager for offboard setpoint ownership. Map the shared yaw heading in degrees to `TrajectorySetpoint.yaw` radians; leave position, acceleration, and yaw-rate fields unused.
- **Invalid obstacle data:** Treat a non-finite or malformed CM5 obstacle reading as unsafe and output zero velocity. A finite reading below the stop threshold still gets the explicit bounded backoff.
- **CM5 command bounds:** Treat the Mac packet as untrusted at the vehicle boundary: valid component-wise limits are enforced locally, and invalid commands fail to zero while a valid close obstacle reading retains authority to back off.

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
- **2026-07-28** — Added `sim/offboard.py` and the deterministic `sim/offboard_control.py` velocity profile. The script primes PX4 with a zero setpoint, streams at 20 Hz, moves north at 0.5 m/s for four seconds, returns to zero velocity, stops offboard, and lands. Standard-library unit tests cover the profile boundaries and safety default. PX4 SITL verified the full connection/arm/offboard/land sequence and reported a measured peak north velocity of 0.52 m/s. A feature branch checkpoint could not be created because this environment exposes `.git` as read-only.
- **2026-07-28** — Added a dependency-free reactive state-machine core. Cognitive intent selects the state, while a forward obstacle nearer than 0.6 m overrides it to `AVOIDING` and commands a slow north-frame backoff. This establishes the safety authority rule: reactive safety may override cognitive intent, but the cognitive layer still cannot bypass reactive velocity generation.
- **2026-07-28** — Routed the live `sim/offboard.py` MAVSDK adapter through `ReactiveController`; the adapter now converts the demo cognitive intent into `FOLLOWING`/`HOVERING` states and never constructs flight velocity commands outside the reactive layer. SITL re-verification preserved the measured 0.52 m/s north velocity and clean landing sequence.
- **2026-07-28** — Added `control/tracking.py`, a dependency-free constant-velocity Kalman tracker for image-plane person detections. It smooths noisy center measurements, rejects out-of-order timestamps, and predicts 300 ms ahead; 5 focused tracker tests plus a repeated-motion check pass. Kept this local rather than adding `filterpy` because the project only needs this small fixed model and avoiding a runtime dependency keeps the Mac/CM5 boundary simpler.
- **2026-07-28** — Added `vision/person_detector.py`, a lazy YOLOv8n adapter that filters class 0 persons, applies a confidence threshold, selects the strongest detection, and emits tracker-compatible center observations. Added `ultralytics` to the Mac-side requirements. Fourteen unit tests pass; the real `ultralytics 8.4.108` runtime loaded `yolov8n.pt` and detected a person in its bundled sample image at 0.836 confidence. Model weights are ignored as local runtime artifacts.
- **2026-07-28** — Added `vision/pipeline.py` to compose detection and tracking into one frame-processing boundary. Missing detections return no target explicitly, while tracker timestamp errors propagate for diagnosis. Seventeen tests pass, and the real YOLOv8n-to-Kalman pipeline produced a tracked center and 300 ms prediction on the bundled person image.
- **2026-07-28** — Bounded intermittent-vision behavior: `PersonTracker.predict()` bridges detector gaps for at most 0.5 seconds, then expires the target; `PersonVisionPipeline` uses this fallback instead of silently dropping every missed frame. Twenty-one tests pass, including gap recovery and stale-target expiry, and the real YOLOv8n first-frame smoke check still passes.
- **2026-07-28** — Added `vision/video_stream.py`, a Mac-side GStreamer RTP/H.264 receiver that emits timestamped BGR frames through a bounded raw pipe. Twenty-five tests pass, and an elevated local GStreamer `videotestsrc` produced and consumed a complete 36-byte 4×3 BGR frame through the same pipe contract. The Wi-Fi milestone remains unchecked until a CM5 sender and real network stream are available.
- **2026-07-28** — Added the voice pipeline: `voice/transcriber.py` wraps faster-whisper, `voice/intent.py` conservatively maps transcripts to cognitive `State` values, and `voice/pipeline.py` connects them without bypassing reactive control. Thirty tests pass; faster-whisper 1.2.1 loaded the tiny.en model and successfully transcribed generated silence. Live microphone capture and the always-on versus push-to-talk choice remain open.
- **2026-07-28** — Added `voice/recorder.py` for fixed-length push-to-talk mono 16 kHz capture and declared `sounddevice`. Thirty-two tests pass; the runtime dependency installed, but this environment exposes zero audio input devices, so physical microphone capture remains unverified.
- **2026-07-28** — Routed `sim/offboard_control.py` through `VoiceCommandPipeline` using deterministic demo transcripts (`follow me` then `hover`). Thirty-three tests pass, and PX4 SITL verified the voice-derived state path through arming, offboard motion at 0.55 m/s observed north velocity, landing, and disarm. The broader full integration milestone remains pending real video/audio sources.
- **2026-07-28** — Hardened `ReactiveController`: `FOLLOWING` now requires a target estimate no older than 0.5 seconds; missing or stale person evidence produces zero velocity, while obstacle backoff remains authoritative. Thirty-five tests pass, and SITL regression preserved 0.55 m/s motion with explicit fresh-target evidence and a clean landing.
- **2026-07-28** — Added `control/runtime.py` as the shared coordinator for cognitive state, timestamped vision targets, obstacle distance, and reactive commands. Refactored `sim/offboard.py` to use it with deterministic voice and target inputs. Forty tests pass, and SITL verified the coordinator path with 0.54 m/s observed north velocity and clean landing; real camera/audio integration remains pending hardware sources.
- **2026-07-28** — Extended the SITL profile with a deterministic forward TOF event from 2–3 seconds at 0.5 m. The scenario now requires and observed `-0.13 m/s` north backoff telemetry after printing `Obstacle detected; backing off.`, then landed cleanly. The single-sensor scope question remains open for hardware design.
- **2026-07-28** — Added an explicit CM5 sender command to `H264StreamConfig`: libcamera → low-latency x264 → RTP/H.264 → UDP. Forty-three tests pass; the Mac host has `x264enc` and `rtph264pay`, but no `libcamerasrc`, so sender execution and the Wi-Fi milestone remain hardware/DEXI-OS validation tasks.
- **2026-07-28** — Added `control/step.py`, the synchronous frame-to-command seam: vision processes a frame, cognitive intent is applied, target freshness and obstacle distance enter `CompanionRuntime`, and one reactive velocity command leaves the stack. Forty-six tests pass, and real YOLOv8n inference on the bundled person image produced the expected `0.5 m/s` reactive command.
- **2026-07-28** — Refactored `sim/offboard.py` to use `CompanionControlStep` with a synthetic vision provider, voice-derived intent, and simulated TOF input. SITL re-verification observed `0.53 m/s` forward and `-0.14 m/s` obstacle backoff velocities, then landed cleanly; 46 tests remain green.
- **2026-07-28** — Added `control/following.py`: visual target height now commands bounded north/backward distance correction and predicted horizontal error commands bounded east correction. Detector observations carry bounding-box size through the Kalman tracker. Fifty tests pass; real YOLOv8n output produced `-0.500 m/s` north and `0.276 m/s` east on the bundled sample, confirming size and position affect the reactive command.
- **2026-07-28** — Visual-follow SITL regression passed through the shared control step: synthetic target geometry produced `0.33 m/s` forward telemetry, the simulated obstacle produced `-0.17 m/s` backoff, and the vehicle landed cleanly. Hardware calibration remains required before trusting camera-to-NED alignment in flight.
- **2026-07-28** — Added a latched `SetpointWatchdog` with a 150 ms deadline; missed deadlines emit zero velocity and remain tripped until the flight loop is safely stopped. The watchdog-protected visual-follow/obstacle SITL run stayed healthy at 20 Hz, observed `0.34 m/s` forward and `-0.17 m/s` backoff, and landed cleanly. Fifty-four tests pass.
- **2026-07-28** — Added `PushToTalkVoicePipeline.listen_once()` to compose recorder output, Whisper transcription, and conservative intent parsing into one optional cognitive state. Unknown speech remains a no-op; fifty-six focused tests pass. Physical microphone capture is still unavailable in this environment.
- **2026-07-28** — Extracted `CompanionControlLoop` as the production one-tick boundary: each frame, intent, and obstacle reading is converted to one reactive command and passed through the latched setpoint watchdog exactly once. Fifty-nine tests pass, and the refactored visual-follow/obstacle SITL run observed `0.31 m/s` forward and `-0.18 m/s` backoff velocities before a clean landing.
- **2026-07-28** — Fixed the RTP/H.264 receiver negotiation by inserting `h264parse` between depayloading and decoding, then added `sim/video_loopback.py`. The reusable verifier passed with a real local GStreamer sender and production receiver, decoding a `(48, 64, 3)` BGR frame. CM5 camera and Wi-Fi validation remain pending hardware.
- **2026-07-28** — Added the reusable offboard frame-reader seam and `sim/offboard_video.py`. The full simulation path now starts a local RTP/H.264 sender, decodes frames through the production receiver, runs the vision/tracker seam, applies deterministic voice intent and TOF obstacle input, and sends the resulting velocity through the watchdog into PX4 SITL. The integrated run observed `0.15 m/s` forward and `-0.21 m/s` obstacle backoff velocities, then landed cleanly; real YOLO/Whisper runtime inputs and CM5/Wi-Fi remain separate hardware-facing validation tasks.
- **2026-07-28** — Added `AsyncLatestFrameReader` so a blocking GStreamer pipe cannot starve the reactive setpoint loop. Empty samples use tracker prediction, and capture timestamps are normalized monotonically before entering the watchdog. Sixty-four tests pass; the clean RTP/SITL regression observed `0.26 m/s` forward and `-0.14 m/s` obstacle backoff without a watchdog trip, then landed cleanly. A timestamp-order regression and an orphaned MAVSDK helper were both diagnosed and fixed at their sources.
- **2026-07-28** — Added `onboard/video_sender.py`, a managed CM5 `libcamerasrc`→RTP/H.264 process with explicit lifecycle and CLI configuration. Sixty-seven tests pass, the sender CLI help is verified, and the existing local RTP loopback still decodes a `(48, 64, 3)` BGR frame. Actual CM5 camera execution remains hardware/DEXI-OS gated because this Mac has no `libcamerasrc` plugin.
- **2026-07-28** — Re-audited the 72-test suite. It remains fast and mostly behavior-focused, but repeated exact motion assertions and a few adapter-call assertions could resist future refactoring. Clarified that test count is not progress, redundant or obsolete tests should be deleted, tunable behavior should use bounds and tolerances, and SITL/Gazebo remains the final authority for flight behavior.
- **2026-07-28** — Added `onboard/safety.py`, the transport-independent CM5 safety envelope. Fresh Mac commands pass through, stale or absent commands become zero velocity, and a local forward TOF reading overrides a fresh command with the bounded backoff. Seventy-two tests pass, and the existing offboard SITL regression observed `0.32 m/s` forward and `-0.17 m/s` backoff before a clean landing. The remaining hardware task is wiring this envelope to the DEXI-OS MAVLink/PX4 forwarding process.
- **2026-07-28** — Added `control/command_packet.py` and connected packet receipt to the CM5 safety envelope. The dependency-free version-1 codec validates finite velocity fields, bounds packet size, and prevents reordered packets from refreshing the local heartbeat. Seventy-eight tests pass; the remaining hardware task is wiring packet reception and safe-command forwarding into DEXI-OS.
- **2026-07-28** — Added `onboard/command_receiver.py` and `sim/command_loopback.py`. The non-blocking UDP receiver drains Mac packets, ignores malformed/reordered data, and exposes only envelope-approved velocity; the real localhost verifier passed fresh `0.3 m/s` forwarding and `-0.2 m/s` obstacle override. Eighty-one tests pass; Wi-Fi, CM5, and PX4-forwarding validation remain hardware-gated.
- **2026-07-28** — Added `onboard/velocity_forwarder.py` and `sim/offboard_udp.py`. PX4 SITL verified the complete packet path through the CM5 safety receiver and MAVSDK NED sink: `0.30 m/s` forward telemetry, `-0.17 m/s` obstacle backoff, and a clean landing. Eighty-two tests pass; the sink is explicitly a SITL stand-in for the eventual DEXI-OS MAVLink forwarder.
- **2026-07-28** — Extended `sim/offboard_udp.py` with a deliberate 0.5-second command-link dropout while the Mac loop continues generating commands. The CM5 envelope expired the missing stream to zero after 150 ms, then resumed safely; clean SITL telemetry observed `0.25 m/s` forward and `-0.15 m/s` obstacle backoff before landing. The Wi-Fi milestone remains unchecked because this is a localhost loss simulation, not a CM5 radio test.
- **2026-07-28** — Fixed reactive state recovery: obstacle safety now temporarily exposes `AVOIDING` without destroying the cognitive intent, and the next clear command restores `FOLLOWING`/`HOVERING`/other saved intent. Eighty-three tests pass; the UDP dropout/obstacle SITL regression observed `0.32 m/s` forward and `-0.19 m/s` backoff before a clean landing.
- **2026-07-28** — Added `onboard/command_service.py` as the minimal CM5 lifecycle boundary: it binds the UDP receiver, polls the local safety envelope at 50 Hz, forwards only approved NED velocity, and emits zero on shutdown. The real localhost loopback verified fresh forwarding, obstacle override, and shutdown zero; 85 tests pass. The first service-backed SITL attempt exposed and fixed a sender-before-bind race, and a clean retry verified command dropout expiry, `0.33 m/s` forward, `-0.17 m/s` obstacle backoff, and a clean landing.
- **2026-07-28** — Inspected the public DroneBlocks DEXI ROS 2 repository and identified the CM5 transport as Micro-ROS serial on `/dev/ttyAMA2` at 3 Mbps into PX4 ROS 2 topics. Added `onboard/ros2_forwarder.py`, a dependency-free adapter that publishes velocity-only offboard heartbeat/setpoints and preserves the shared yaw-heading contract. Eighty-six tests pass; `rclpy` node composition and hardware serial validation remain gated on DEXI-OS bring-up.
- **2026-07-28** — Audited the same DEXI repository for TOF ingress and found no public distance driver or ROS topic; kept the hardware sensor boundary unresolved rather than guessing GPIO/I²C details. Hardened `OnboardSafetyEnvelope` so malformed, infinite, or `NaN` obstacle readings fail to zero instead of being interpreted as clear. Eighty-seven tests pass.
- **2026-07-28** — Added CM5-side command bounds independent of Mac behavior: horizontal NED components are limited to `±0.5 m/s`, vertical to `±0.3 m/s`, and any non-finite command field fails to zero. Boundary and rejection tests pass; the existing SITL profile remains within these limits.
- **2026-07-28** — Strengthened `sim.offboard_udp` with an adversarial `1.0 m/s` wire command. PX4 SITL confirmed the CM5 envelope suppressed the out-of-bounds command while preserving heartbeat expiry, obstacle backoff, and clean landing; telemetry observed `0.31 m/s` forward and `-0.18 m/s` backoff.
- **2026-07-28** — Made CM5 startup ordering explicit with `SafetyCommandService.start()`: both UDP verifiers bind synchronously before creating their async service task, eliminating the sender-before-bind race by construction. Eighty-nine tests pass; real UDP loopback and adversarial/dropout/obstacle PX4 SITL both completed with clean landing.
- **2026-07-28** — Reconciled the project guide's Tech Stack and Open Questions with the implemented dependency-free tracker/state machine and resolved push-to-talk decision, removing obsolete `filterpy`/`transitions` and always-on voice ambiguity.
- **2026-07-28** — Restored the explicit milestone/checkpoint reread rule after a guide consistency audit found it had been accidentally dropped from the active instructions.
- **2026-07-28** — Tightened `onboard.ros2_forwarder.Ros2VelocityForwarder`: PX4 `OffboardControlMode` now receives the same injected timestamp as its trajectory setpoint, and every non-velocity control flag is explicitly false. The focused adapter test and full 89-test suite pass; ROS 2 node composition remains hardware-gated.
- **2026-07-28** — Hardened the CM5 obstacle boundary against boolean sensor values. Because Python booleans are numeric subclasses, `True`/`False` previously bypassed the malformed-reading check; both now fail safe to zero alongside `NaN`, infinity, and non-numeric values.
- **2026-07-28** — Mirrored malformed-obstacle fail-safe handling in `ReactiveController`: `NaN`, infinity, booleans, and non-numeric readings now produce zero velocity without entering `AVOIDING` or changing the saved cognitive intent. The CM5 remains an independent final safety boundary.
- **2026-07-28** — Hardened `VisualFollower` against malformed detector geometry and configuration. Non-finite, boolean, or non-numeric target size/position now holds zero, and follow speed/configuration values must be finite and non-negative; valid visual-follow behavior remains bounded.
- **2026-07-28** — Validated `PersonTracker` measurements and timing before they enter Kalman state. Non-finite, boolean, out-of-range confidence, negative geometry, and invalid tracker configuration now raise a clear `ValueError` before state mutation; normal tracking and short-gap prediction remain unchanged.
- **2026-07-28** — Made `SafetyCommandService` cleanup unconditional when the hardware forwarder fails: the service still attempts one zero command, but always closes the UDP receiver afterward. Added a lifecycle regression test for a failed forwarder.
- **2026-07-28** — Tightened voice intent parsing to exact token phrases and reject negation, substring lookalikes, and non-string transcripts. Ambiguous or negative speech remains a no-op instead of changing cognitive flight state.
- **2026-07-28** — Bound `VisualFollower` configuration to the CM5 envelope (`0.5 m/s` forward and `0.3 m/s` lateral), preventing Mac-side tuning from emitting commands that the vehicle boundary would reject wholesale.
- **2026-07-28** — Hardened `sim.offboard` teardown: frame/model failures now produce a zero shutdown setpoint, always stop offboard, land the vehicle, and only then re-raise the original error. Normal voice/video SITL behavior remains unchanged.
- **2026-07-28** — Removed the misleading Unix-clock fallback from `Ros2VelocityForwarder`; the DEXI ROS node must inject its vehicle/ROS timestamp source explicitly, preventing silently invalid PX4 timestamps on hardware.
- **2026-07-28** — Completed the offboard teardown hardening: failures while sending the final zero or stopping offboard are now recorded, cleanup continues, and landing still runs before the failure is reported.
- **2026-07-28** — Added `onboard.ros2_bridge.Ros2SafetyBridge` and `python -m onboard.ros2_bridge`: the DEXI-side composition now binds the UDP receiver before a dedicated safety thread, injects the ROS/PX4 clock, subscribes to the standard distance-sensor output, and starts with an unsafe/no-reading value. The fake-node lifecycle test and full 98-test suite pass; ROS 2 and physical serial/sensor validation remain hardware-gated.
- **2026-07-28** — Matched the bridge's ROS entry point to the DEXI/PX4 DDS QoS contract: best-effort, transient-local, keep-last depth 1 is now supplied explicitly to both PX4 input publishers and the distance-sensor subscription; Mac-side tests continue to inject a fake profile without importing ROS.
- **2026-07-28** — Made the bridge QoS profile mandatory at construction, eliminating an accidental default that could silently create incompatible PX4 DDS endpoints outside the DEXI entry point.
- **2026-07-28** — Audited the public DEXI offboard manager lifecycle and documented the ownership boundary: `onboard.ros2_bridge` must not run alongside `px4_offboard_manager`, whose hold/navigation states publish position-mode PX4 setpoints that would conflict with the companion's velocity-only contract.
- **2026-07-28** — Reconciled the CM5 service documentation with the implemented bridge: the service is now transport-independent, while `onboard.ros2_bridge` is the concrete DEXI-OS forwarder composition rather than a future placeholder.
- **2026-07-28** — Added a 150 ms freshness timeout to `LatestDistanceSensor`: initial, malformed, and silent/stale ROS distance data now fail through the existing CM5 envelope to zero instead of preserving an old clear reading. Clock-injected tests cover fresh and expired readings.
- **2026-07-28** — Moved H.264 stream port/dimension/framerate validation into the shared `H264StreamConfig`, so the CM5 sender and Mac receiver reject invalid transport formats at the same boundary.
- **2026-07-28** — Centralized GStreamer child cleanup: sender and receiver now wait at most two seconds after terminate, then kill a stuck process before releasing it, preventing media shutdown from blocking the safety loop indefinitely.
- **2026-07-28** — Hardened `YoloPersonDetector` at the model boundary: invalid thresholds and malformed/non-finite/reversed boxes are rejected as no detection before tracker state or reactive velocity can be affected.
- **2026-07-28** — Re-ran the real YOLOv8n detector and production detector-to-Kalman pipeline on the bundled `bus.jpg`: a person was detected at `0.865` confidence and yielded a valid track prediction. The full suite now contains 102 passing tests.
- **2026-07-28** — Added `sim.video_yolo_loopback`: a real JPEG is repeatedly sent through the RTP/H.264 transport, decoded at `640x480`, processed by production YOLOv8n plus the Kalman tracker, and converted into a bounded reactive velocity command. The verifier passed with a `203 px` person target and `(-0.346, -0.089) m/s` NED command; the full suite now contains 104 passing tests. Hardware Wi-Fi camera validation remains open.
