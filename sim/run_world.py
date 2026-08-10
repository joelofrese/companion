"""Run one PX4/Gazebo scenario and clean up."""

import argparse
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

BOOT_MARKER = "pxh>"
BOOT_MARKER_BYTES = BOOT_MARKER.encode()
BOOT_TIMEOUT_S = 120.0
BOOT_RETRIES = 1
SHUTDOWN_TIMEOUT_S = 10.0
OLLAMA_BOOT_TIMEOUT_S = 10.0
# Keep the PX4 and Gazebo starting headings aligned.
DEFAULT_MODEL_POSE = "0,0,0,0,0,0"


class _BootError(RuntimeError):
    """A PX4 process failed before its shell became ready."""


def _read_output(process, ready, finished):
    try:
        tail = b""
        while chunk := process.stdout.read(4096):
            if not ready.is_set() and BOOT_MARKER_BYTES in tail + chunk:
                ready.set()
            if not ready.is_set():
                print(f"[PX4] {chunk.decode(errors='replace')}", end="", flush=True)
            tail = (tail + chunk)[-(len(BOOT_MARKER_BYTES) - 1):]
    finally:
        finished.set()


def _stop_process_group(process, process_group_id):
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stop_process(process):
    """Stop one process that owns its own process group."""

    if process is None:
        return
    try:
        process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    _stop_process_group(process, process_group_id)


def _stop_processes(processes):
    """Stop processes in reverse start order."""

    for process in reversed(processes):
        _stop_process(process)


def _validate_pose(pose: Optional[str]) -> Optional[str]:
    if pose is None:
        return None
    values = pose.split(",")
    if len(values) != 6:
        raise RuntimeError("model pose must contain x,y,z,roll,pitch,yaw")
    try:
        numbers = [float(value) for value in values]
    except ValueError as error:
        raise RuntimeError("model pose must contain six numbers") from error
    if not all(math.isfinite(value) for value in numbers):
        raise RuntimeError("model pose must contain finite numbers")
    return ",".join(str(value) for value in numbers)


def _start_ollama_if_needed():
    """Return a local Ollama process only when this run had to start it."""

    from control.ollama_client import OllamaClient

    client = OllamaClient(timeout_s=1.0)
    try:
        client.check()
        return None
    except RuntimeError:
        pass
    ollama = shutil.which("ollama")
    if ollama is None:
        raise RuntimeError("Ollama is unavailable and `ollama` is not installed")
    process = subprocess.Popen(
        [ollama, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + OLLAMA_BOOT_TIMEOUT_S
    while process.poll() is None and time.monotonic() < deadline:
        try:
            client.check()
        except RuntimeError:
            time.sleep(0.1)
        else:
            print("Started local Ollama for this simulation.")
            return process
    _stop_process(process)
    raise RuntimeError("Ollama did not become ready")


def _stop_ollama(process, *models):
    """Release models and stop only a server started by this run."""

    if process is None:
        return
    from control.ollama_client import OllamaClient

    client = OllamaClient(timeout_s=1.0)
    for model in set(models):
        try:
            client.unload(model)
        except RuntimeError:
            pass
    _stop_process(process)


def _run_once(
    px4_dir: Path,
    companion_dir: Path,
    world_file: Path,
    image_path: Optional[Path],
    expect_person: bool,
    world: str,
    stdbuf: str,
    exploratory: bool,
    faults: bool,
    camera: bool,
    depth: bool,
    duration_s: Optional[float],
    ollama: bool,
    vlm_model: str,
    llm_model: str,
    ollama_timeout: float,
    initial_intent: str,
    model_pose: Optional[str],
    memory_path: Optional[Path],
    snapshot_path: Optional[Path],
    dialogue_request: Optional[str],
    trace: bool,
    moving_person: bool,
    headless: bool,
) -> int:
    environment = os.environ.copy()
    environment["PX4_GZ_WORLD"] = world
    environment["PX4_GZ_MODEL_POSE"] = model_pose or DEFAULT_MODEL_POSE
    # Start Gazebo here so every run uses the same user-controlled camera.
    environment["PX4_GZ_STANDALONE"] = "1"
    environment["PX4_GZ_NO_FOLLOW"] = "1"
    environment["GZ_IP"] = "127.0.0.1"
    local_worlds = companion_dir / "sim/worlds"
    world_processes = []
    gz = shutil.which("gz")
    if gz is None:
        raise RuntimeError("gz is required for Gazebo simulation")
    models = px4_dir / "Tools/simulation/gz/models"
    stock_worlds = px4_dir / "Tools/simulation/gz/worlds"
    plugins = px4_dir / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
    resource_path = environment.get("GZ_SIM_RESOURCE_PATH")
    environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
        path for path in (str(local_worlds), str(models), str(stock_worlds), resource_path)
        if path
    )
    environment["GZ_SIM_SYSTEM_PLUGIN_PATH"] = os.pathsep.join(
        path
        for path in (str(plugins), environment.get("GZ_SIM_SYSTEM_PLUGIN_PATH"))
        if path
    )
    environment["GZ_SIM_SERVER_CONFIG_PATH"] = str(
        px4_dir / "src/modules/simulation/gz_bridge/server.config"
    )
    gui_config = companion_dir / "sim/gazebo_gui.config"
    try:
        world_processes.append(
            subprocess.Popen(
                [gz, "sim", "-r", "-s", str(world_file)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        )
    except BaseException:
        _stop_processes(world_processes)
        raise
    model = "gz_x500"
    if depth:
        model = "gz_x500_depth"
    elif camera:
        model = "gz_x500_mono_cam"

    process = None
    try:
        process = subprocess.Popen(
            [
                stdbuf,
                "-oL",
                "-eL",
                "make",
                "px4_sitl",
                model,
            ],
            cwd=px4_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
            env=environment,
        )
        process_group_id = os.getpgid(process.pid)
    except BaseException:
        _stop_process(process)
        _stop_processes(world_processes)
        raise
    ready = threading.Event()
    finished = threading.Event()
    output_thread = threading.Thread(
        target=_read_output,
        args=(process, ready, finished),
        daemon=True,
    )
    output_thread.start()
    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while not ready.is_set():
            if finished.wait(0.2):
                raise _BootError(
                    f"PX4 exited before pxh> (code {process.poll()})"
                )
            if time.monotonic() >= deadline:
                raise _BootError("PX4 did not reach pxh> before the boot timeout")
        if not headless and not environment.get("HEADLESS"):
            # Open the GUI after PX4 has spawned the model. This keeps Gazebo
            # from reframing or tracking the vehicle as it appears.
            world_processes.append(
                subprocess.Popen(
                    [gz, "sim", "-g", "--gui-config", str(gui_config)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=environment,
                )
            )
        scenario = [sys.executable, "-u", "-m"]
        if image_path:
            scenario += ["sim.offboard_full", str(image_path)]
            if expect_person:
                scenario.append("--expect-person")
        else:
            scenario += ["sim.world"]
            if exploratory:
                scenario.append("--explore")
            if faults:
                scenario.append("--faults")
            if camera:
                scenario.append("--camera")
            if depth:
                scenario.append("--depth")
            if moving_person:
                scenario.append("--moving-person")
            if exploratory:
                scenario += ["--intent", initial_intent]
            if ollama:
                scenario += [
                    "--ollama",
                    "--vlm-model",
                    vlm_model,
                    "--llm-model",
                    llm_model,
                    "--ollama-timeout",
                    str(ollama_timeout),
                ]
            if memory_path is not None:
                scenario += ["--memory", str(memory_path)]
            if snapshot_path is not None:
                scenario += ["--snapshot", str(snapshot_path)]
            if dialogue_request is not None:
                scenario += ["--request", dialogue_request]
            if trace:
                scenario.append("--trace")
            scenario += ["--world", world]
            if duration_s is not None:
                scenario += ["--duration", str(duration_s)]
        result = subprocess.run(scenario, cwd=companion_dir, check=False)
        return result.returncode
    finally:
        _stop_process_group(process, process_group_id)
        _stop_processes(world_processes)


def run(
    px4_dir: Path,
    companion_dir: Path,
    image_path: Optional[Path] = None,
    expect_person: bool = False,
    world: Optional[str] = None,
    exploratory: bool = False,
    faults: bool = False,
    camera: bool = False,
    depth: bool = False,
    duration_s: Optional[float] = None,
    ollama: bool = False,
    vlm_model: str = "moondream",
    llm_model: str = "moondream",
    ollama_timeout: float = 60.0,
    initial_intent: str = "explore the surroundings",
    model_pose: Optional[str] = None,
    memory_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    dialogue_request: Optional[str] = None,
    trace: bool = False,
    moving_person: bool = False,
    headless: bool = False,
) -> int:
    if world is None:
        world = "objects" if exploratory and (camera or depth) else "default"
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise RuntimeError("stdbuf is required to observe PX4 boot output")
    if not px4_dir.is_dir():
        raise RuntimeError(f"PX4 directory does not exist: {px4_dir}")
    if image_path is not None and not image_path.is_file():
        raise RuntimeError(f"scenario image does not exist: {image_path}")
    if exploratory and image_path is not None:
        raise RuntimeError("exploratory simulation cannot use an RTP image")
    if faults and not exploratory:
        raise RuntimeError("fault injection requires exploratory simulation")
    if camera and image_path is not None:
        raise RuntimeError("camera mode cannot use an RTP image")
    if depth and image_path is not None:
        raise RuntimeError("depth mode cannot use an RTP image")
    if not isinstance(initial_intent, str) or not initial_intent.strip():
        raise RuntimeError("initial intent must be a non-empty string")
    if camera and depth:
        raise RuntimeError("camera and depth modes cannot run together")
    if camera and not exploratory:
        raise RuntimeError("Gazebo camera mode requires exploratory simulation")
    if depth and not exploratory:
        raise RuntimeError("Gazebo depth mode requires exploratory simulation")
    if exploratory and world != "default" and not (camera or depth):
        raise RuntimeError(
            "a non-default exploratory world requires --camera or --depth"
        )
    if moving_person and not (
        exploratory and world == "objects" and (camera or depth)
    ):
        raise RuntimeError(
            "moving-person simulation requires exploratory objects camera or depth mode"
        )
    if faults and camera:
        raise RuntimeError("fault injection requires synthetic safety or depth mode")
    if memory_path is not None and not exploratory:
        raise RuntimeError("experience memory requires exploratory simulation")
    if memory_path is not None and image_path is not None:
        raise RuntimeError("experience memory cannot run with an RTP image scenario")
    if snapshot_path is not None and image_path is not None:
        raise RuntimeError("camera snapshot cannot use an RTP image scenario")
    if snapshot_path is not None and not (camera or depth):
        raise RuntimeError("camera snapshot requires Gazebo camera or depth mode")
    if dialogue_request is not None and not exploratory:
        raise RuntimeError("dialogue request requires exploratory simulation")
    if dialogue_request is not None and image_path is not None:
        raise RuntimeError("dialogue request cannot use an RTP image scenario")
    if trace and image_path is not None:
        raise RuntimeError("brain trace requires a synthetic world scenario")
    if dialogue_request is not None and not dialogue_request.strip():
        raise RuntimeError("dialogue request must not be empty")
    if ollama and not (exploratory and (camera or depth)):
        raise RuntimeError("Ollama simulation requires exploratory camera or depth mode")
    if duration_s is not None and image_path is not None:
        raise RuntimeError("simulation duration cannot use an RTP image scenario")
    if expect_person and image_path is None:
        raise RuntimeError("expect-person requires an RTP image scenario")
    if duration_s is not None and (duration_s <= 0.0 or not math.isfinite(duration_s)):
        raise RuntimeError("simulation duration must be positive")
    model_pose = _validate_pose(model_pose)
    world_file = companion_dir / "sim/worlds" / f"{world}.sdf"
    if not world_file.is_file():
        world_file = px4_dir / "Tools/simulation/gz/worlds" / f"{world}.sdf"
    if not world_file.is_file():
        raise RuntimeError(f"Gazebo world does not exist: {world_file}")

    ollama_process = _start_ollama_if_needed() if ollama else None
    try:
        for attempt in range(BOOT_RETRIES + 1):
            try:
                return _run_once(
                    px4_dir=px4_dir,
                    companion_dir=companion_dir,
                    world_file=world_file,
                    image_path=image_path,
                    expect_person=expect_person,
                    world=world,
                    stdbuf=stdbuf,
                    exploratory=exploratory,
                    faults=faults,
                    camera=camera,
                    depth=depth,
                    duration_s=duration_s,
                    ollama=ollama,
                    vlm_model=vlm_model,
                    llm_model=llm_model,
                    ollama_timeout=ollama_timeout,
                    initial_intent=initial_intent,
                    model_pose=model_pose,
                    memory_path=memory_path,
                    snapshot_path=snapshot_path,
                    dialogue_request=dialogue_request,
                    trace=trace,
                    moving_person=moving_person,
                    headless=headless,
                )
            except _BootError:
                if attempt == BOOT_RETRIES:
                    raise
                print("PX4 did not boot; retrying once.", file=sys.stderr)
    finally:
        _stop_ollama(ollama_process, vlm_model, llm_model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a complete PX4/Gazebo companion scenario")
    parser.add_argument(
        "--px4-dir",
        type=Path,
        default=Path.home() / "Code/Croppie/PX4-Autopilot",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="run the RTP image full-stack verification with this image",
    )
    parser.add_argument(
        "--expect-person",
        action="store_true",
        help="use the deterministic person fixture for the RTP check",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="run the synthetic world with live dialogue and observation-only behavior",
    )
    parser.add_argument(
        "--faults",
        action="store_true",
        help="inject the normal safety and link faults into an exploratory run",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="use Gazebo's x500_mono_cam and feed its rendered frames to the Mac brain",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="use Gazebo's x500_depth camera and feed its depth readings to CM5 safety",
    )
    parser.add_argument(
        "--moving-person",
        action="store_true",
        help="move the visual mannequin through the objects world",
    )
    parser.add_argument(
        "--intent",
        default="explore the surroundings",
        help="initial high-level intent for an exploratory run",
    )
    parser.add_argument(
        "--pose",
        help="PX4 spawn pose as x,y,z,roll,pitch,yaw",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="use local Ollama VLM and LLM for an exploratory camera or depth run",
    )
    parser.add_argument("--vlm-model", default="moondream")
    parser.add_argument("--llm-model", default="moondream")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
    parser.add_argument(
        "--memory",
        type=Path,
        help="persist conscious experience across exploratory runs",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="save the first Gazebo camera frame for visual inspection",
    )
    parser.add_argument(
        "--request",
        help="send one dialogue request automatically during an exploratory run",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print meaningful VLM observations, conscious decisions, and command reasons",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening the Gazebo GUI",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="world simulation duration in seconds (default: 32)",
    )
    parser.add_argument(
        "--world",
        help="Gazebo world name (exploratory runs default to objects)",
    )
    args = parser.parse_args(argv)
    try:
        return run(
            px4_dir=args.px4_dir.expanduser().resolve(),
            companion_dir=Path(__file__).resolve().parent.parent,
            image_path=args.image.expanduser().resolve() if args.image else None,
            expect_person=args.expect_person,
            world=args.world,
            exploratory=args.explore,
            faults=args.faults,
            camera=args.camera,
            depth=args.depth,
            duration_s=args.duration,
            ollama=args.ollama,
            vlm_model=args.vlm_model,
            llm_model=args.llm_model,
            ollama_timeout=args.ollama_timeout,
            initial_intent=args.intent,
            model_pose=args.pose,
            memory_path=args.memory.expanduser().resolve() if args.memory else None,
            snapshot_path=args.snapshot.expanduser().resolve() if args.snapshot else None,
            dialogue_request=args.request,
            trace=args.trace,
            moving_person=args.moving_person,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"simulation runner: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
