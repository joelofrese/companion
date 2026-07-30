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


def _run_once(
    px4_dir: Path,
    companion_dir: Path,
    image_path: Optional[Path],
    expect_person: bool,
    world: str,
    stdbuf: str,
    exploratory: bool,
    camera: bool,
    lidar: bool,
    duration_s: Optional[float],
    ollama: bool,
    vlm_model: str,
    llm_model: str,
    ollama_timeout: float,
) -> int:
    environment = os.environ.copy()
    environment["PX4_GZ_WORLD"] = world
    model = "gz_x500"
    if camera:
        model = "gz_x500_mono_cam"
    elif lidar:
        model = "gz_x500_lidar_front"

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
        scenario = [sys.executable, "-u", "-m"]
        if image_path:
            scenario += ["sim.offboard_full", str(image_path)]
            if expect_person:
                scenario.append("--expect-person")
        else:
            scenario += ["sim.world"]
            if exploratory:
                scenario.append("--explore")
            if camera:
                scenario.append("--camera")
            if lidar:
                scenario.append("--lidar")
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
            scenario += ["--world", world]
            if duration_s is not None:
                scenario += ["--duration", str(duration_s)]
        result = subprocess.run(scenario, cwd=companion_dir, check=False)
        return result.returncode
    finally:
        _stop_process_group(process, process_group_id)


def run(
    px4_dir: Path,
    companion_dir: Path,
    image_path: Optional[Path] = None,
    expect_person: bool = False,
    world: str = "default",
    exploratory: bool = False,
    camera: bool = False,
    lidar: bool = False,
    duration_s: Optional[float] = None,
    ollama: bool = False,
    vlm_model: str = "gemma3:4b",
    llm_model: str = "gemma3:4b",
    ollama_timeout: float = 60.0,
) -> int:
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise RuntimeError("stdbuf is required to observe PX4 boot output")
    if not px4_dir.is_dir():
        raise RuntimeError(f"PX4 directory does not exist: {px4_dir}")
    if image_path is not None and not image_path.is_file():
        raise RuntimeError(f"scenario image does not exist: {image_path}")
    if camera and lidar:
        raise RuntimeError("camera and lidar simulation modes cannot run together")
    if lidar and not exploratory:
        raise RuntimeError("Gazebo lidar mode requires exploratory simulation")
    if duration_s is not None and (duration_s <= 0.0 or not math.isfinite(duration_s)):
        raise RuntimeError("simulation duration must be positive")
    world_file = px4_dir / "Tools/simulation/gz/worlds" / f"{world}.sdf"
    if not world_file.is_file():
        raise RuntimeError(f"Gazebo world does not exist: {world_file}")

    for attempt in range(BOOT_RETRIES + 1):
        try:
            return _run_once(
                px4_dir,
                companion_dir,
                image_path,
                expect_person,
                world,
                stdbuf,
                exploratory,
                camera,
                lidar,
                duration_s,
                ollama,
                vlm_model,
                llm_model,
                ollama_timeout,
            )
        except _BootError:
            if attempt == BOOT_RETRIES:
                raise
            print("PX4 did not boot; retrying once.", file=sys.stderr)


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
        help="require the image to produce following and lateral motion",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="run the synthetic world with live dialogue and observation-only behavior",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="use Gazebo's x500_mono_cam and feed its rendered frames to the Mac brain",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="use Gazebo's x500_lidar_front and feed its range readings to CM5 safety",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="use local Ollama VLM and LLM for an exploratory camera run",
    )
    parser.add_argument("--vlm-model", default="gemma3:4b")
    parser.add_argument("--llm-model", default="gemma3:4b")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
    parser.add_argument(
        "--duration",
        type=float,
        help="world simulation duration in seconds (default: 32)",
    )
    parser.add_argument("--world", default="default", help="Gazebo world name from PX4")
    args = parser.parse_args(argv)
    if args.explore and args.image:
        parser.error("--explore cannot be combined with --image")
    if args.camera and args.image:
        parser.error("--camera cannot be combined with --image")
    if args.lidar and args.image:
        parser.error("--lidar cannot be combined with --image")
    if args.camera and args.lidar:
        parser.error("--camera and --lidar cannot be combined")
    if args.camera and not args.explore:
        parser.error("--camera requires --explore")
    if args.lidar and not args.explore:
        parser.error("--lidar requires --explore")
    if args.ollama and not args.camera:
        parser.error("--ollama requires --camera")
    if args.duration is not None and args.image:
        parser.error("--duration cannot be combined with --image")
    if args.expect_person and not args.image:
        parser.error("--expect-person requires --image")
    if args.duration is not None and (args.duration <= 0.0 or not math.isfinite(args.duration)):
        parser.error("--duration must be positive")
    try:
        return run(
            args.px4_dir.expanduser().resolve(),
            Path(__file__).resolve().parent.parent,
            args.image.expanduser().resolve() if args.image else None,
            args.expect_person,
            args.world,
            args.explore,
            args.camera,
            args.lidar,
            args.duration,
            args.ollama,
            args.vlm_model,
            args.llm_model,
            args.ollama_timeout,
        )
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"simulation runner: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
