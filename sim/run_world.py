"""Run one PX4/Gazebo scenario and clean up."""

import argparse
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
    world: str,
    stdbuf: str,
) -> int:
    environment = os.environ.copy()
    environment["PX4_GZ_WORLD"] = world

    process = subprocess.Popen(
        [stdbuf, "-oL", "-eL", "make", "px4_sitl", "gz_x500"],
        cwd=px4_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
                raise RuntimeError(
                    f"PX4 exited before pxh> (code {process.poll()})"
                )
            if time.monotonic() >= deadline:
                raise RuntimeError("PX4 did not reach pxh> before the boot timeout")
        scenario = [sys.executable, "-u", "-m"]
        scenario += ["sim.offboard_full", str(image_path)] if image_path else ["sim.world"]
        result = subprocess.run(scenario, cwd=companion_dir, check=False)
        return result.returncode
    finally:
        _stop_process_group(process, process_group_id)


def run(
    px4_dir: Path,
    companion_dir: Path,
    image_path: Optional[Path] = None,
    world: str = "default",
) -> int:
    stdbuf = shutil.which("stdbuf")
    if stdbuf is None:
        raise RuntimeError("stdbuf is required to observe PX4 boot output")
    if not px4_dir.is_dir():
        raise RuntimeError(f"PX4 directory does not exist: {px4_dir}")
    if image_path is not None and not image_path.is_file():
        raise RuntimeError(f"scenario image does not exist: {image_path}")
    world_file = px4_dir / "Tools/simulation/gz/worlds" / f"{world}.sdf"
    if not world_file.is_file():
        raise RuntimeError(f"Gazebo world does not exist: {world_file}")

    for attempt in range(BOOT_RETRIES + 1):
        try:
            return _run_once(px4_dir, companion_dir, image_path, world, stdbuf)
        except RuntimeError:
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
        help="run production RTP/YOLO full-stack verification with this person image",
    )
    parser.add_argument("--world", default="default", help="Gazebo world name from PX4")
    args = parser.parse_args(argv)
    try:
        return run(
            args.px4_dir.expanduser().resolve(),
            Path(__file__).resolve().parent.parent,
            args.image.expanduser().resolve() if args.image else None,
            args.world,
        )
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"simulation runner: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
